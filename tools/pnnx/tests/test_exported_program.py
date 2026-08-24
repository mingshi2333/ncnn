#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import torch


PNNX = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class BufferLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        self.register_buffer("weight", weight)

    def forward(self, x):
        return torch.relu(torch.nn.functional.linear(x, self.weight))


class NonPersistentBufferLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        return torch.relu(torch.nn.functional.linear(x, self.weight))


class TensorConstantLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10

    def forward(self, x):
        return torch.relu(torch.nn.functional.linear(x, self.weight))


class SharedStorageLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        self.first_weight = torch.nn.Parameter(storage[:3, :], requires_grad=False)
        self.second_weight = torch.nn.Parameter(
            storage.reshape(-1)[:6].reshape(2, 3), requires_grad=False
        )

    def forward(self, x):
        x = torch.relu(torch.nn.functional.linear(x, self.first_weight))
        return torch.relu(torch.nn.functional.linear(x, self.second_weight))


def run_pnnx(work_dir, model_path):
    return subprocess.run(
        [str(PNNX), model_path.name],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def save_exported_program(model, archive_path):
    exported_program = torch.export.export(model, (torch.ones(2, 4),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(exported_program, archive_path)


def expected_output(model):
    torch.manual_seed(0)
    return model(torch.rand(2, 4))


def rewrite_archive(source_path, destination_path, mutate):
    with zipfile.ZipFile(source_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    mutate(entries)

    with zipfile.ZipFile(
        destination_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as destination:
        for name, data in entries.items():
            destination.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def rewrite_payload_configs(source_path, destination_path, mutate):
    def mutate_entries(entries):
        weights_path = next(
            name for name in entries if name.endswith("_weights_config.json")
        )
        constants_path = next(
            name for name in entries if name.endswith("_constants_config.json")
        )
        weights_document = json.loads(entries[weights_path])
        constants_document = json.loads(entries[constants_path])
        mutate(weights_document["config"], constants_document["config"])
        entries[weights_path] = json.dumps(
            weights_document, separators=(",", ":")
        ).encode()
        entries[constants_path] = json.dumps(
            constants_document, separators=(",", ":")
        ).encode()

    rewrite_archive(source_path, destination_path, mutate_entries)


def load_generated_output(work_dir, basename):
    module_path = work_dir / f"{basename}_pnnx.py"
    spec = importlib.util.spec_from_file_location(
        f"test_exported_program_{basename}_pnnx", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated module {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        spec.loader.exec_module(module)
        return module.test_inference()
    finally:
        os.chdir(previous_work_dir)


class ExportedProgramEndToEndTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel().eval()

        torch.manual_seed(0)
        self.expected = self.model(torch.rand(2, 4))

    def assert_conversion_matches(self, work_dir, model_path, expected=None):
        result = run_pnnx(work_dir, model_path)
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode(errors="replace"),
        )

        actual = load_generated_output(work_dir, model_path.stem)
        if expected is None:
            expected = self.expected
        self.assertTrue(
            torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
            f"generated output mismatch\nexpected={expected}\nactual={actual}",
        )

    def assert_conversion_fails(self, work_dir, model_path, *messages):
        result = run_pnnx(work_dir, model_path)
        self.assertNotEqual(result.returncode, 0, "conversion unexpectedly succeeded")

        stderr = result.stderr.decode(errors="replace")
        for message in messages:
            self.assertIn(message, stderr)

        self.assertFalse((work_dir / f"{model_path.stem}.pnnx.param").exists())
        self.assertFalse((work_dir / f"{model_path.stem}.pnnx.bin").exists())
        self.assertFalse((work_dir / f"{model_path.stem}_pnnx.py").exists())

    def test_exported_program_routes_by_archive_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            pt2_path = work_dir / "tiny.pt2"
            renamed_path = work_dir / "tiny_renamed.bin"

            save_exported_program(self.model, pt2_path)
            shutil.copyfile(pt2_path, renamed_path)

            self.assert_conversion_matches(work_dir, pt2_path)
            self.assert_conversion_matches(work_dir, renamed_path)

    def test_torchscript_with_pt2_suffix_still_uses_torchscript_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            torchscript_path = work_dir / "legacy.pt2"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traced = torch.jit.trace(self.model, (torch.ones(2, 4),))
            traced.save(str(torchscript_path))

            self.assert_conversion_matches(work_dir, torchscript_path)

    def test_persistent_buffer_uses_weights_payload(self):
        model = BufferLinearModel().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "buffer.pt2"
            save_exported_program(model, archive_path)

            self.assert_conversion_matches(
                work_dir, archive_path, expected_output(model)
            )

    def test_non_persistent_buffer_uses_constants_payload(self):
        model = NonPersistentBufferLinearModel().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "non_persistent_buffer.pt2"
            save_exported_program(model, archive_path)

            self.assert_conversion_matches(
                work_dir, archive_path, expected_output(model)
            )

    def test_tensor_constant_uses_constants_payload(self):
        model = TensorConstantLinearModel().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "constant.pt2"
            save_exported_program(model, archive_path)

            self.assert_conversion_matches(
                work_dir, archive_path, expected_output(model)
            )

    def test_payload_record_in_wrong_config_is_rejected(self):
        cases = (
            (
                TinyModel().eval(),
                "linear.weight",
                "weights",
                "parameter linear.weight is present in constants config",
            ),
            (
                BufferLinearModel().eval(),
                "weight",
                "weights",
                "buffer weight is present in constants config",
            ),
            (
                NonPersistentBufferLinearModel().eval(),
                "weight",
                "constants",
                "buffer weight is present in weights config",
            ),
            (
                TensorConstantLinearModel().eval(),
                "weight",
                "constants",
                "tensor constant weight is present in weights config",
            ),
        )

        for index, (model, target, expected_config, message) in enumerate(cases):
            with self.subTest(target=target, expected_config=expected_config):
                with tempfile.TemporaryDirectory() as temp_dir:
                    work_dir = Path(temp_dir)
                    source_path = work_dir / f"wrong_config_source_{index}.pt2"
                    mutated_path = work_dir / f"wrong_config_{index}.pt2"
                    save_exported_program(model, source_path)

                    def duplicate_into_wrong_config(weights, constants):
                        if expected_config == "weights":
                            constants[target] = dict(weights[target])
                        else:
                            weights[target] = dict(constants[target])

                    rewrite_payload_configs(
                        source_path, mutated_path, duplicate_into_wrong_config
                    )
                    self.assert_conversion_fails(work_dir, mutated_path, message)

    def test_payload_is_param_matches_signature_kind(self):
        cases = (
            (
                TinyModel().eval(),
                "linear.weight",
                "weights",
                False,
                "parameter linear.weight in weights config has is_param=false",
            ),
            (
                BufferLinearModel().eval(),
                "weight",
                "weights",
                True,
                "buffer weight in weights config has is_param=true",
            ),
            (
                NonPersistentBufferLinearModel().eval(),
                "weight",
                "constants",
                True,
                "buffer weight in constants config has is_param=true",
            ),
            (
                TensorConstantLinearModel().eval(),
                "weight",
                "constants",
                True,
                "tensor constant weight in constants config has is_param=true",
            ),
        )

        for index, (model, target, config_name, value, message) in enumerate(cases):
            with self.subTest(target=target, config=config_name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    work_dir = Path(temp_dir)
                    source_path = work_dir / f"is_param_source_{index}.pt2"
                    mutated_path = work_dir / f"is_param_{index}.pt2"
                    save_exported_program(model, source_path)

                    def change_is_param(weights, constants):
                        config = weights if config_name == "weights" else constants
                        config[target]["is_param"] = value

                    rewrite_payload_configs(source_path, mutated_path, change_is_param)
                    self.assert_conversion_fails(work_dir, mutated_path, message)

    def test_pickled_payload_reports_fqn_and_archive_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "pickle_source.pt2"
            mutated_path = work_dir / "pickle.pt2"
            save_exported_program(self.model, source_path)
            payload_path_name = ""

            def mark_pickled(weights, constants):
                nonlocal payload_path_name
                entry = weights["linear.weight"]
                payload_path_name = entry["path_name"]
                entry["use_pickle"] = True

            rewrite_payload_configs(source_path, mutated_path, mark_pickled)
            self.assert_conversion_fails(
                work_dir,
                mutated_path,
                "pickled payload is unsupported",
                "parameter linear.weight",
                payload_path_name,
            )

    def test_schema_errors_name_the_archive_entry(self):
        cases = (
            (
                "_weights_config.json",
                b'{"config":[]}',
                "invalid exported payload config",
                "$.config: expected object",
            ),
            (
                "/models/",
                b"[]",
                "invalid exported program",
                "$: expected object",
            ),
        )

        for index, (entry_marker, replacement, prefix, detail) in enumerate(cases):
            with self.subTest(entry_marker=entry_marker):
                with tempfile.TemporaryDirectory() as temp_dir:
                    work_dir = Path(temp_dir)
                    source_path = work_dir / f"schema_source_{index}.pt2"
                    mutated_path = work_dir / f"schema_{index}.pt2"
                    save_exported_program(self.model, source_path)
                    mutated_entry = ""

                    def corrupt_entry(entries):
                        nonlocal mutated_entry
                        mutated_entry = next(
                            name
                            for name in entries
                            if entry_marker in name and name.endswith(".json")
                        )
                        entries[mutated_entry] = replacement

                    rewrite_archive(source_path, mutated_path, corrupt_entry)
                    self.assert_conversion_fails(
                        work_dir, mutated_path, prefix, mutated_entry, detail
                    )

    def test_missing_payload_reports_full_archive_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "missing_blob_source.pt2"
            mutated_path = work_dir / "missing_blob.pt2"
            save_exported_program(self.model, source_path)
            removed_path = ""

            def remove_weight_blob(entries):
                nonlocal removed_path
                weights_path = next(
                    name for name in entries if name.endswith("_weights_config.json")
                )
                config = json.loads(entries[weights_path])["config"]
                path_name = config["linear.weight"]["path_name"]
                root = weights_path.split("/data/weights/", 1)[0]
                removed_path = f"{root}/data/weights/{path_name}"
                del entries[removed_path]

            rewrite_archive(source_path, mutated_path, remove_weight_blob)
            self.assert_conversion_fails(
                work_dir, mutated_path, removed_path, "is missing"
            )

    def test_truncated_payload_reports_fqn_and_archive_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "truncated_blob_source.pt2"
            mutated_path = work_dir / "truncated_blob.pt2"
            save_exported_program(self.model, source_path)
            truncated_path = ""

            def truncate_weight_blob(entries):
                nonlocal truncated_path
                weights_path = next(
                    name for name in entries if name.endswith("_weights_config.json")
                )
                config = json.loads(entries[weights_path])["config"]
                path_name = config["linear.weight"]["path_name"]
                root = weights_path.split("/data/weights/", 1)[0]
                truncated_path = f"{root}/data/weights/{path_name}"
                entries[truncated_path] = entries[truncated_path][:4]

            rewrite_archive(source_path, mutated_path, truncate_weight_blob)
            self.assert_conversion_fails(
                work_dir,
                mutated_path,
                "linear.weight",
                truncated_path,
                "tensor view exceeds storage",
            )

    def test_payload_shape_must_match_graph_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "shape_source.pt2"
            mutated_path = work_dir / "shape.pt2"
            save_exported_program(self.model, source_path)

            def transpose_weight_shape(weights, constants):
                tensor_meta = weights["linear.weight"]["tensor_meta"]
                tensor_meta["sizes"] = [{"as_int": 4}, {"as_int": 3}]
                tensor_meta["strides"] = [{"as_int": 3}, {"as_int": 1}]

            rewrite_payload_configs(
                source_path, mutated_path, transpose_weight_shape
            )
            self.assert_conversion_fails(
                work_dir,
                mutated_path,
                "parameter linear.weight shape does not match tensor metadata",
            )

    def test_shared_parameter_storage_views_match_eager(self):
        model = SharedStorageLinearModel().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "shared_storage.pt2"
            save_exported_program(model, archive_path)

            with zipfile.ZipFile(archive_path, "r") as archive:
                weights_path = next(
                    name
                    for name in archive.namelist()
                    if name.endswith("_weights_config.json")
                )
                config = json.loads(archive.read(weights_path))["config"]
            self.assertEqual(
                config["first_weight"]["path_name"],
                config["second_weight"]["path_name"],
                "fixture no longer shares one serialized storage blob",
            )

            self.assert_conversion_matches(
                work_dir, archive_path, expected_output(model)
            )


if __name__ == "__main__":
    unittest.main()
