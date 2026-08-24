#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCHEMA_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(*arguments):
    return subprocess.run(
        [str(SCHEMA_HELPER), *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TensorConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.tensor([1.0, 2.0, 3.0])

    def forward(self, x):
        return x + self.value


class BufferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        return x + self.value


class NonPersistentBufferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "value", torch.tensor([1.0, 2.0, 3.0]), persistent=False
        )

    def forward(self, x):
        return x + self.value


class ConstantInputModel(torch.nn.Module):
    def forward(self, x, count: int):
        return x + count


class PrimitiveConstantInputsModel(torch.nn.Module):
    def forward(
        self,
        x,
        count: int,
        scale: float,
        enabled: bool,
        label: str,
        missing,
    ):
        if enabled and label == "ok" and missing is None:
            return x * scale + count
        return x


class NonFiniteFloatModel(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x, -float("inf"))


class ExportedProgramSchemaTest(unittest.TestCase):
    def export_and_inspect(self, model, inputs, command="inspect"):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "schema.pt2"
            exported_program = torch.export.export(model.eval(), inputs)
            torch.export.save(exported_program, archive_path)
            return run_helper(command, archive_path)

    def test_typed_schema_unit_matrix(self):
        result = run_helper()

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"exported program schema tests passed", result.stdout)

    def test_real_torch_export_header_and_payload_configs(self):
        result = self.export_and_inspect(TinyModel(), (torch.ones(2, 4),))

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|2|0",
                "payload|weights|linear.bias|weight_1|1|0|1|7|3|1|0|cpu|null|7|1",
                "payload|weights|linear.weight|weight_0|1|0|1|7|3,4|4,1|0|cpu|null|7|1",
            ],
        )

    def test_real_torch_export_graph_and_signature(self):
        result = self.export_and_inspect(
            TinyModel(), (torch.ones(2, 4),), command="inspect-graph"
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|2|0",
                "graph|3|2|1|5|0|3|1",
                "input|parameter|p_linear_weight|linear.weight|0",
                "input|parameter|p_linear_bias|linear.bias|0",
                "input|user_input|x||0",
                "node|linear|torch.ops.aten.linear.default|3|1",
                "node|relu|torch.ops.aten.relu.default|1|1",
                "output|user_output|relu|",
                "payload|weights|linear.bias|weight_1|1|0|1|7|3|1|0|cpu|null|7|1",
                "payload|weights|linear.weight|weight_0|1|0|1|7|3,4|4,1|0|cpu|null|7|1",
            ],
        )

    def test_real_tensor_constant_config(self):
        result = self.export_and_inspect(TensorConstantModel(), (torch.ones(3),))

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|0|1",
                "payload|constants|value|tensor_0|0|0|1|7|3|1|0|cpu|null|7|0",
            ],
        )

    def test_real_persistent_buffer_config(self):
        result = self.export_and_inspect(BufferModel(), (torch.ones(3),))

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|1|0",
                "payload|weights|value|weight_0|0|0|1|7|3|1|0|cpu|null|7|0",
            ],
        )

    def test_real_non_persistent_buffer_config(self):
        result = self.export_and_inspect(
            NonPersistentBufferModel(), (torch.ones(3),)
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|0|1",
                "payload|constants|value|tensor_0|0|0|1|7|3|1|0|cpu|null|7|0",
            ],
        )

    def test_real_constant_input_signature(self):
        result = self.export_and_inspect(
            ConstantInputModel(), (torch.ones(3), 7), command="inspect-graph"
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(
            result.stdout.decode().splitlines(),
            [
                f"header|8|20|{torch.__version__}|10|0|0",
                "graph|2|1|1|2|0|2|1",
                "input|user_input|x||0",
                "input|constant_input|count||0",
                "node|add|torch.ops.aten.add.Tensor|2|1",
                "output|user_output|add|",
            ],
        )

    def test_real_primitive_constant_input_signatures(self):
        result = self.export_and_inspect(
            PrimitiveConstantInputsModel(),
            (torch.ones(3), 7, 1.5, True, "ok", None),
            command="inspect-graph",
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        lines = result.stdout.decode().splitlines()
        self.assertIn("input|constant_input|count||0", lines)
        self.assertIn("input|constant_input|scale||0", lines)
        self.assertIn("input|constant_input|enabled||0", lines)
        self.assertIn("input|constant_input|label||0", lines)
        self.assertIn("input|constant_input|missing||0", lines)

    def test_real_nonfinite_float_argument(self):
        result = self.export_and_inspect(
            NonFiniteFloatModel(), (torch.ones(3),), command="inspect-graph"
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(
            "node|full_like|torch.ops.aten.full_like.default|3|1",
            result.stdout.decode().splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
