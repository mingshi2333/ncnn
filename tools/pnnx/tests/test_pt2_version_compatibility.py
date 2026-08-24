#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import torch


PNNX = Path(sys.argv[1]).resolve()
OPERATOR_HELPER = Path(sys.argv[2]).resolve()
sys.argv = [sys.argv[0]]

EXPECTED_SCHEMA_MINOR = {
    (2, 8): 8,
    (2, 9): 14,
    (2, 10): 15,
    (2, 11): 17,
    (2, 12): 20,
    (2, 13): 20,
}


def torch_version_tuple():
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch.__version__)
    if match is None:
        raise AssertionError("cannot parse torch version %r" % torch.__version__)
    return tuple(int(value or 0) for value in match.groups())


def save_exported_program(model, example_inputs, archive_path):
    exported_program = torch.export.export(model.eval(), example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(exported_program, archive_path)


def read_model_document(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        model_paths = [
            name for name in archive.namelist() if name.endswith("/models/model.json")
        ]
        if len(model_paths) != 1:
            raise AssertionError("expected one model.json, found %r" % model_paths)
        return json.loads(archive.read(model_paths[0]))


def rewrite_model_document(source_path, destination_path, mutate):
    with zipfile.ZipFile(source_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    model_paths = [name for name in entries if name.endswith("/models/model.json")]
    if len(model_paths) != 1:
        raise AssertionError("expected one model.json, found %r" % model_paths)

    model_path = model_paths[0]
    document = json.loads(entries[model_path])
    mutate(document)
    entries[model_path] = json.dumps(document, separators=(",", ":")).encode()

    with zipfile.ZipFile(
        destination_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as destination:
        for name, data in entries.items():
            destination.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def run_pnnx(work_dir, archive_path):
    return subprocess.run(
        [str(PNNX), archive_path.name],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_operator_helper(archive_path):
    return subprocess.run(
        [str(OPERATOR_HELPER), "inspect", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_generated_output(work_dir, basename):
    module_path = work_dir / (basename + "_pnnx.py")
    spec = importlib.util.spec_from_file_location(
        "test_pt2_version_compatibility_" + basename, module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load generated module %s" % module_path)

    module = importlib.util.module_from_spec(spec)
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        spec.loader.exec_module(module)
        return module.test_inference()
    finally:
        os.chdir(previous_work_dir)


def parse_node_lines(output):
    nodes = {}
    for line in output.decode().splitlines():
        fields = line.split("|", 4)
        if len(fields) != 5 or fields[0] != "node":
            raise AssertionError("unexpected helper output: %s" % line)
        nodes[fields[1]] = fields[4].split(";") if fields[4] else []
    return nodes


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


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


class Pt2VersionCompatibilityTest(unittest.TestCase):
    def require_pt2_producer(self):
        version = torch_version_tuple()
        if version < (2, 8, 0):
            self.skipTest("torch.export PT2 archive starts at the PyTorch 2.8 CI row")
        return version

    def assert_rejected(self, work_dir, archive_path, message):
        result = run_pnnx(work_dir, archive_path)
        self.assertNotEqual(result.returncode, 0, "conversion unexpectedly succeeded")
        self.assertIn(message, result.stderr.decode(errors="replace"))
        self.assertFalse((work_dir / (archive_path.stem + ".pnnx.param")).exists())
        self.assertFalse((work_dir / (archive_path.stem + ".pnnx.bin")).exists())
        self.assertFalse((work_dir / (archive_path.stem + "_pnnx.py")).exists())

    def test_real_producer_schema_and_tiny_conversion(self):
        version = self.require_pt2_producer()
        producer = version[:2]
        self.assertIn(producer, EXPECTED_SCHEMA_MINOR)

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "tiny.pt2"

            torch.manual_seed(42)
            model = TinyModel().eval()
            save_exported_program(model, (torch.ones(2, 4),), archive_path)
            document = read_model_document(archive_path)

            self.assertEqual(document["schema_version"]["major"], 8)
            self.assertEqual(
                document["schema_version"]["minor"],
                EXPECTED_SCHEMA_MINOR[producer],
            )
            self.assertIsInstance(document["opset_version"]["aten"], int)
            print(
                "producer=%s schema=%d.%d aten=%d"
                % (
                    torch.__version__,
                    document["schema_version"]["major"],
                    document["schema_version"]["minor"],
                    document["opset_version"]["aten"],
                )
            )

            if producer == (2, 8):
                self.assert_rejected(
                    work_dir,
                    archive_path,
                    "PyTorch 2.8 legacy pickled-payload PT2 is unsupported",
                )
                return

            self.assertLessEqual(version, (2, 13, 0))
            result = run_pnnx(work_dir, archive_path)
            self.assertEqual(
                result.returncode, 0, result.stderr.decode(errors="replace")
            )

            torch.manual_seed(0)
            expected = model(torch.rand(2, 4))
            actual = load_generated_output(work_dir, archive_path.stem)
            self.assertTrue(
                torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
                "generated output mismatch\nexpected=%s\nactual=%s"
                % (expected, actual),
            )

    def test_real_default_argument_schema(self):
        version = self.require_pt2_producer()
        if version < (2, 9, 0) or version > (2, 13, 0):
            self.skipTest("default argument compatibility covers raw-payload producers")

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "defaults.pt2"
            save_exported_program(
                OmittedDefaultsModel(), (torch.ones(1, 1, 5, 5),), archive_path
            )
            result = run_operator_helper(archive_path)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        nodes = parse_node_lines(result.stdout)
        self.assertEqual(
            nodes["torch.ops.aten.add.Tensor"],
            ["self=tensor:x", "other=tensor:x", "alpha=int:1"],
        )
        self.assertEqual(
            nodes["torch.ops.aten.flatten.using_ints"],
            ["self=tensor:add", "start_dim=int:1", "end_dim=int:-1"],
        )
        self.assertEqual(
            nodes["torch.ops.aten.conv2d.default"],
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

    def test_rejects_unverified_and_malformed_headers(self):
        version = self.require_pt2_producer()
        if version < (2, 9, 0):
            self.skipTest("PyTorch 2.8 is rejected from its real legacy payload layout")

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "source.pt2"
            save_exported_program(TinyModel(), (torch.ones(2, 4),), source_path)

            cases = [
                (
                    "future.pt2",
                    lambda document: document.update(
                        torch_version="2.14.0",
                        schema_version={"major": 8, "minor": 21},
                    ),
                    "untested torch producer version",
                ),
                (
                    "future_patch.pt2",
                    lambda document: document.update(
                        torch_version="2.12.2",
                        schema_version={"major": 8, "minor": 20},
                    ),
                    "untested torch producer version",
                ),
                (
                    "current_future_patch.pt2",
                    lambda document: document.update(
                        torch_version="2.13.1",
                        schema_version={"major": 8, "minor": 20},
                    ),
                    "untested torch producer version",
                ),
                (
                    "schema_major.pt2",
                    lambda document: document.update(
                        torch_version="2.12.1",
                        schema_version={"major": 9, "minor": 20},
                    ),
                    "incompatible schema major",
                ),
                (
                    "missing_opset.pt2",
                    lambda document: document.update(
                        torch_version="2.12.1",
                        schema_version={"major": 8, "minor": 20},
                        opset_version={},
                    ),
                    "$.opset_version.aten: missing required field",
                ),
            ]

            for filename, mutate, diagnostic in cases:
                archive_path = work_dir / filename
                rewrite_model_document(source_path, archive_path, mutate)
                with self.subTest(filename=filename):
                    self.assert_rejected(work_dir, archive_path, diagnostic)


if __name__ == "__main__":
    unittest.main()
