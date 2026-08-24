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


IR_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(*arguments):
    return subprocess.run(
        [str(IR_HELPER), *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class GRUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.GRU(4, 3, batch_first=True)

    def forward(self, x, hidden):
        return self.recurrent(x, hidden)


class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.LSTM(4, 3, batch_first=True)

    def forward(self, x, hidden, cell):
        return self.recurrent(x, (hidden, cell))


class RNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.RNN(4, 3, nonlinearity="relu", batch_first=True)

    def forward(self, x, hidden):
        return self.recurrent(x, hidden)


class ExportedProgramIRTest(unittest.TestCase):
    def test_ir_unit_matrix(self):
        result = run_helper()

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"exported program ir tests passed", result.stdout)

    def test_real_linear_relu_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "linear_relu.pt2"
            exported_program = torch.export.export(
                LinearReluModel().eval(), (torch.ones(2, 4),)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.export.save(exported_program, archive_path)

            result = run_helper("inspect", archive_path)

        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        lines = result.stdout.decode().splitlines()
        self.assertEqual(
            lines,
            [
                "before|pnnx.Attribute|linear.weight||p_linear_weight",
                "before|pnnx.Attribute|linear.bias||p_linear_bias",
                "before|pnnx.Input|pnnx_input_0||x",
                "before|aten::linear|linear|x,p_linear_weight,p_linear_bias|linear",
                "before|aten::relu|relu|linear|relu",
                "before|pnnx.Output|pnnx_output_0|relu|",
                "after-types|pnnx.Attribute,pnnx.Attribute,pnnx.Input,F.linear,F.relu,pnnx.Output",
            ],
        )

    def test_real_recurrent_graphs(self):
        cases = (
            ("gru", GRUModel(), (torch.ones(2, 5, 4), torch.ones(1, 2, 3)), "nn.GRU"),
            (
                "lstm",
                LSTMModel(),
                (torch.ones(2, 5, 4), torch.ones(1, 2, 3), torch.ones(1, 2, 3)),
                "nn.LSTM",
            ),
            ("rnn", RNNModel(), (torch.ones(2, 5, 4), torch.ones(1, 2, 3)), "nn.RNN"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for name, model, inputs, expected_type in cases:
                with self.subTest(name=name):
                    archive_path = Path(temp_dir) / f"{name}.pt2"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        exported_program = torch.export.export(model.eval(), inputs)
                        torch.export.save(exported_program, archive_path)

                    result = run_helper("inspect", archive_path)

                    self.assertEqual(
                        result.returncode, 0, result.stderr.decode(errors="replace")
                    )
                    after_types = result.stdout.decode().splitlines()[-1]
                    self.assertIn(expected_type, after_types)


if __name__ == "__main__":
    unittest.main()
