#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch


TENSOR_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(*arguments):
    return subprocess.run(
        [str(TENSOR_HELPER), *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def expected_hex(tensor):
    return tensor.detach().contiguous().numpy().tobytes().hex()


def parse_tensor_lines(output):
    result = {}
    for line in output.decode().splitlines():
        fields = line.split("|")
        if len(fields) != 7 or fields[0] != "tensor":
            raise AssertionError(f"unexpected helper output: {line}")
        result[(fields[1], fields[2])] = fields[3:]
    return result


class ViewWeightsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.transposed = torch.nn.Parameter(base.t(), requires_grad=False)

        offset_storage = torch.arange(10, dtype=torch.float32)
        self.offset = torch.nn.Parameter(offset_storage[3:7], requires_grad=False)

    def forward(self, x):
        return x + self.transposed.sum() + self.offset.sum()


class SharedStorageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(8, dtype=torch.float32)
        self.left = torch.nn.Parameter(storage[:5], requires_grad=False)
        self.right = torch.nn.Parameter(storage[2:7], requires_grad=False)

    def forward(self, x):
        return x + self.left.sum() + self.right.sum()


class EmptyAndScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.empty = torch.nn.Parameter(torch.empty(0, 3), requires_grad=False)
        self.scalar = torch.nn.Parameter(torch.tensor(2.5), requires_grad=False)

    def forward(self, x):
        return x + self.empty.sum() + self.scalar


class TensorConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.arange(6, dtype=torch.int64).reshape(2, 3).t()

    def forward(self, x):
        return x + self.value.sum()


class ExportedProgramTensorTest(unittest.TestCase):
    def export_and_inspect(self, model):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "tensor.pt2"
            exported_program = torch.export.export(model.eval(), (torch.ones(1),))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.export.save(exported_program, archive_path)
            return run_helper("inspect", archive_path)

    def test_materializer_unit_matrix(self):
        result = run_helper()

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"exported program tensor tests passed", result.stdout)

    def test_real_transposed_and_offset_weights(self):
        model = ViewWeightsModel()
        result = self.export_and_inspect(model)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        tensors = parse_tensor_lines(result.stdout)
        self.assertEqual(tensors[("weights", "transposed")][1], "1")
        self.assertEqual(tensors[("weights", "transposed")][2], "4,3")
        self.assertEqual(
            tensors[("weights", "transposed")][3], expected_hex(model.transposed)
        )
        self.assertEqual(tensors[("weights", "offset")][1], "1")
        self.assertEqual(tensors[("weights", "offset")][2], "4")
        self.assertEqual(tensors[("weights", "offset")][3], expected_hex(model.offset))

    def test_real_shared_storage_views(self):
        model = SharedStorageModel()
        result = self.export_and_inspect(model)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        tensors = parse_tensor_lines(result.stdout)
        left = tensors[("weights", "left")]
        right = tensors[("weights", "right")]
        self.assertEqual(left[0], right[0])
        self.assertEqual(left[3], expected_hex(model.left))
        self.assertEqual(right[3], expected_hex(model.right))

    def test_real_empty_and_scalar_weights(self):
        model = EmptyAndScalarModel()
        result = self.export_and_inspect(model)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        tensors = parse_tensor_lines(result.stdout)
        self.assertEqual(tensors[("weights", "empty")][2], "0,3")
        self.assertEqual(tensors[("weights", "empty")][3], "")
        self.assertEqual(tensors[("weights", "scalar")][2], "")
        self.assertEqual(tensors[("weights", "scalar")][3], expected_hex(model.scalar))

    def test_real_noncontiguous_tensor_constant(self):
        model = TensorConstantModel()
        result = self.export_and_inspect(model)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        tensors = parse_tensor_lines(result.stdout)
        self.assertEqual(tensors[("constants", "value")][1], "5")
        self.assertEqual(tensors[("constants", "value")][2], "3,2")
        self.assertEqual(tensors[("constants", "value")][3], expected_hex(model.value))


if __name__ == "__main__":
    unittest.main()
