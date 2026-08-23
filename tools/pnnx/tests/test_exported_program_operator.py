#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import torch


OPERATOR_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(*arguments):
    return subprocess.run(
        [str(OPERATOR_HELPER), *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_node_lines(output):
    result = {}
    for line in output.decode().splitlines():
        fields = line.split("|", 5)
        if len(fields) != 6 or fields[0] != "node":
            raise AssertionError(f"unexpected helper output: {line}")
        arguments = fields[5].split(";") if fields[5] else []
        result[fields[1]] = (fields[2], fields[3], fields[4], arguments)
    return result


class OmittedDefaultsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(2, 1, 3, 3), requires_grad=False
        )

    def forward(self, x):
        added = torch.ops.aten.add.Tensor(x, x)
        flattened = torch.ops.aten.flatten.using_ints(added, 1)
        convolved = torch.ops.aten.conv2d.default(
            x, self.weight, None, [1, 1], [0, 0]
        )
        return flattened, convolved


class ExplicitKeywordModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.add.Tensor(x, x, alpha=3)


class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class ExportedProgramOperatorTest(unittest.TestCase):
    def export_and_inspect(self, model, example_input):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "operator.pt2"
            exported_program = torch.export.export(model.eval(), (example_input,))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.export.save(exported_program, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                model_path = next(
                    name
                    for name in archive.namelist()
                    if name.endswith("/models/model.json")
                )
                model_json = json.loads(archive.read(model_path))
            return run_helper("inspect", archive_path), model_json

    def test_operator_unit_matrix(self):
        result = run_helper()

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"exported program operator tests passed", result.stdout)

    def test_real_omitted_defaults(self):
        result, model_json = self.export_and_inspect(
            OmittedDefaultsModel(), torch.ones(1, 1, 5, 5)
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        serialized_nodes = {
            node["target"]: [argument["name"] for argument in node["inputs"]]
            for node in model_json["graph_module"]["graph"]["nodes"]
        }
        self.assertEqual(
            serialized_nodes["torch.ops.aten.add.Tensor"], ["self", "other"]
        )
        self.assertEqual(
            serialized_nodes["torch.ops.aten.flatten.using_ints"],
            ["self", "start_dim"],
        )
        self.assertEqual(
            serialized_nodes["torch.ops.aten.conv2d.default"], ["input", "weight"]
        )
        nodes = parse_node_lines(result.stdout)
        self.assertEqual(
            nodes["torch.ops.aten.add.Tensor"][3],
            ["self=tensor:x", "other=tensor:x", "alpha=int:1"],
        )
        self.assertEqual(
            nodes["torch.ops.aten.flatten.using_ints"][3],
            ["self=tensor:add", "start_dim=int:1", "end_dim=int:-1"],
        )
        self.assertEqual(
            nodes["torch.ops.aten.conv2d.default"][3],
            [
                "input=tensor:x",
                "weight=tensor:p_weight",
                "bias=none",
                "stride=ints:1,1",
                "padding=ints:0,0",
                "dilation=ints:1,1",
                "groups=int:1",
            ],
        )

    def test_real_explicit_keyword(self):
        result, _ = self.export_and_inspect(
            ExplicitKeywordModel(), torch.ones(2, 3)
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        nodes = parse_node_lines(result.stdout)
        add = nodes["torch.ops.aten.add.Tensor"]
        self.assertEqual(add[:3], ("aten::add", "Tensor", "0"))
        self.assertEqual(
            add[3], ["self=tensor:x", "other=tensor:x", "alpha=int:3"]
        )

    def test_real_initial_allowlist(self):
        result, _ = self.export_and_inspect(LinearReluModel(), torch.ones(2, 4))

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        nodes = parse_node_lines(result.stdout)
        linear = nodes["torch.ops.aten.linear.default"]
        relu = nodes["torch.ops.aten.relu.default"]
        self.assertEqual(linear[:3], ("aten::linear", "default", "1"))
        self.assertEqual(relu[:3], ("aten::relu", "default", "1"))
        self.assertEqual(linear[3][-1], "bias=tensor:p_linear_bias")


if __name__ == "__main__":
    unittest.main()
