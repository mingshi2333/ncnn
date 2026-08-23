#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from packaging import version


PNNX = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class DtypeIdentityModel(torch.nn.Module):
    def forward(self, token_ids, values, mask):
        return token_ids, values, mask


def save_exported_program(model, example_inputs, archive_path):
    exported_program = torch.export.export(model, example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(exported_program, archive_path)


def run_pnnx(work_dir, model_path):
    return subprocess.run(
        [str(PNNX), model_path.name],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_generated_module(work_dir, basename):
    module_path = work_dir / f"{basename}_pnnx.py"
    spec = importlib.util.spec_from_file_location(
        f"test_exported_program_roundtrip_{basename}", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated module {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_work_dir)
    return module


def call_in_work_dir(work_dir, function, *args):
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        return function(*args)
    finally:
        os.chdir(previous_work_dir)


@unittest.skipIf(
    version.parse(torch.__version__) < version.parse("2.9"),
    "modern exported program packages require PyTorch 2.9 or newer",
)
class ExportedProgramRoundTripTest(unittest.TestCase):
    def convert(self, work_dir, model_path):
        result = run_pnnx(work_dir, model_path)
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode(errors="replace"),
        )
        return load_generated_module(work_dir, model_path.stem)

    def test_generated_source_builds_typed_tuple_inputs_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "dtypes.pt2"
            example_inputs = (
                torch.ones(2, 3, dtype=torch.int64),
                torch.ones(2, 3, dtype=torch.float64),
                torch.ones(2, 3, dtype=torch.bool),
            )
            save_exported_program(
                DtypeIdentityModel().eval(), example_inputs, archive_path
            )

            module = self.convert(work_dir, archive_path)
            source = (work_dir / "dtypes_pnnx.py").read_text()

            self.assertEqual(source.count("def _create_example_inputs():"), 1)
            self.assertIn("torch.randint(0, 10, (2, 3), dtype=torch.long)", source)
            self.assertIn("torch.rand(2, 3, dtype=torch.double)", source)
            self.assertIn("torch.randint(0, 2, (2, 3), dtype=torch.bool)", source)
            self.assertRegex(
                source,
                r"return \(v_[A-Za-z0-9_]+, v_[A-Za-z0-9_]+, v_[A-Za-z0-9_]+\)",
            )

            generated_inputs = call_in_work_dir(
                work_dir, module._create_example_inputs
            )
            self.assertIsInstance(generated_inputs, tuple)
            self.assertEqual(len(generated_inputs), 3)
            self.assertEqual(generated_inputs[0].dtype, torch.int64)
            self.assertEqual(generated_inputs[1].dtype, torch.float64)
            self.assertEqual(generated_inputs[2].dtype, torch.bool)

            call_in_work_dir(work_dir, module.export_torchscript)
            self.assertTrue((work_dir / "dtypes_pnnx.py.pt").is_file())

    def test_default_export_round_trips_without_overwriting_input(self):
        torch.manual_seed(42)
        model = TinyModel().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "tiny.pt2"
            save_exported_program(model, (torch.ones(2, 4),), archive_path)
            original_archive = archive_path.read_bytes()

            first_module = self.convert(work_dir, archive_path)
            first_source = (work_dir / "tiny_pnnx.py").read_text()
            self.assertIn(
                "def export_exported_program(example_inputs=None):", first_source
            )
            self.assertRegex(first_source, r"return \(v_[A-Za-z0-9_]+,\)")
            self.assertIn("example_inputs = _create_example_inputs()", first_source)

            helper_source = first_source.split(
                "def export_exported_program(example_inputs=None):", 1
            )[1].split("\ndef ", 1)[0]
            self.assertNotIn("net.float()", helper_source)

            call_in_work_dir(work_dir, first_module.export_torchscript)
            self.assertTrue((work_dir / "tiny_pnnx.py.pt").is_file())

            exported_program = call_in_work_dir(
                work_dir, first_module.export_exported_program
            )
            self.assertIsInstance(exported_program, torch.export.ExportedProgram)
            self.assertEqual(archive_path.read_bytes(), original_archive)

            roundtrip_path = work_dir / "tiny_pnnx.pt2"
            self.assertTrue(roundtrip_path.is_file())
            second_module = self.convert(work_dir, roundtrip_path)

            torch.manual_seed(0)
            expected = model(torch.rand(2, 4))
            first_output = call_in_work_dir(work_dir, first_module.test_inference)
            second_output = call_in_work_dir(work_dir, second_module.test_inference)
            self.assertTrue(
                torch.allclose(expected, first_output, rtol=1e-5, atol=1e-5),
                f"first generated output mismatch\nexpected={expected}\nactual={first_output}",
            )
            self.assertTrue(
                torch.allclose(expected, second_output, rtol=1e-5, atol=1e-5),
                f"second generated output mismatch\nexpected={expected}\nactual={second_output}",
            )

    def test_custom_examples_require_and_preserve_a_tuple(self):
        torch.manual_seed(42)
        model = TinyModel().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "custom.pt2"
            save_exported_program(model, (torch.ones(2, 4),), archive_path)
            module = self.convert(work_dir, archive_path)

            with self.assertRaisesRegex(TypeError, "example_inputs must be a tuple"):
                call_in_work_dir(
                    work_dir,
                    module.export_exported_program,
                    torch.ones(3, 4),
                )

            custom_inputs = (torch.full((3, 4), 0.25, dtype=torch.float32),)
            exported_program = call_in_work_dir(
                work_dir, module.export_exported_program, custom_inputs
            )
            self.assertIsInstance(exported_program, torch.export.ExportedProgram)

            generated_model = call_in_work_dir(work_dir, module.Model).eval()
            expected = generated_model(*custom_inputs)
            actual = exported_program.module()(*custom_inputs)
            self.assertEqual(tuple(actual.shape), (3, 3))
            self.assertTrue(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
