#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


PT2_ARCHIVE_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(command, archive_path, entry=None):
    arguments = [str(PT2_ARCHIVE_HELPER), command, str(archive_path)]
    if entry is not None:
        arguments.append(entry)

    return subprocess.run(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def valid_entries(root="random-root", model_name="model", byteorder=b"little"):
    return {
        f"{root}/archive_format": b"pt2",
        f"{root}/archive_version": b"0",
        f"{root}/byteorder": byteorder,
        f"{root}/models/{model_name}.json": b'{"graph":{}}',
        f"{root}/data/weights/{model_name}_weights_config.json": b'{"config":{}}',
        f"{root}/data/constants/{model_name}_constants_config.json": b'{"config":{}}',
    }


def write_archive(path, entries, compressed_entries=()):
    compressed_entries = set(compressed_entries)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in entries.items():
            compression = zipfile.ZIP_DEFLATED if name in compressed_entries else zipfile.ZIP_STORED
            archive.writestr(name, payload, compress_type=compression)


def corrupt_entry_payload(path, entry_name):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(entry_name)

    data = bytearray(path.read_bytes())
    name_length = int.from_bytes(data[info.header_offset + 26 : info.header_offset + 28], "little")
    extra_length = int.from_bytes(data[info.header_offset + 28 : info.header_offset + 30], "little")
    payload_offset = info.header_offset + 30 + name_length + extra_length
    data[payload_offset] ^= 0x01
    path.write_bytes(data)


def expected_layout(root="random-root", model_name="model", byteorder="little"):
    return (
        f"{root}|0|{model_name}|{byteorder}|"
        f"{root}/models/{model_name}.json|"
        f"{root}/data/weights/{model_name}_weights_config.json|"
        f"{root}/data/constants/{model_name}_constants_config.json"
    ).encode()


class TinyModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class Pt2ArchiveTest(unittest.TestCase):
    def write_fixture(self, temp_dir, entries=None, compressed_entries=()):
        archive_path = Path(temp_dir) / "fixture.pt2"
        write_archive(archive_path, valid_entries() if entries is None else entries, compressed_entries)
        return archive_path

    def test_accepts_dynamic_root_and_one_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = self.write_fixture(temp_dir)
            result = run_helper("inspect", archive_path)

            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.rstrip(b"\r\n"), expected_layout())

    def test_reads_little_and_big_byteorder_markers(self):
        for byteorder in ("little", "big"):
            with self.subTest(byteorder=byteorder), tempfile.TemporaryDirectory() as temp_dir:
                entries = valid_entries(byteorder=byteorder.encode())
                archive_path = self.write_fixture(temp_dir, entries)
                result = run_helper("inspect", archive_path)

                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(result.stdout.rstrip(b"\r\n"), expected_layout(byteorder=byteorder))

    def test_rejects_archive_version_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["random-root/archive_version"] = b"1"
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"unsupported pt2 archive version 1", result.stderr)

    def test_reports_aoti_only_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = {
                "random-root/archive_format": b"pt2",
                "random-root/archive_version": b"0",
                "random-root/byteorder": b"little",
                "random-root/data/aotinductor/model/model.so": b"native-code",
            }
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"AOTInductor-only pt2 package is unsupported", result.stderr)

    def test_reports_package_without_exported_program(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = {
                "random-root/archive_format": b"pt2",
                "random-root/archive_version": b"0",
                "random-root/byteorder": b"little",
            }
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"pt2 package contains no ExportedProgram model", result.stderr)

    def test_rejects_two_model_json_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries.update(
                {
                    "random-root/models/second.json": b"{}",
                    "random-root/data/weights/second_weights_config.json": b"{}",
                    "random-root/data/constants/second_constants_config.json": b"{}",
                }
            )
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"multiple ExportedPrograms in one pt2 package are unsupported", result.stderr)

    def test_rejects_missing_weights_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            del entries["random-root/data/weights/model_weights_config.json"]
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"random-root/data/weights/model_weights_config.json is missing", result.stderr)

    def test_rejects_missing_constants_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            del entries["random-root/data/constants/model_constants_config.json"]
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"random-root/data/constants/model_constants_config.json is missing", result.stderr)

    def test_rejects_any_compressed_pt2_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["random-root/extra/unused.bin"] = b"unused"
            archive_path = self.write_fixture(
                temp_dir,
                entries,
                compressed_entries=("random-root/extra/unused.bin",),
            )
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                b"compressed pt2 entry is unsupported random-root/extra/unused.bin",
                result.stderr,
            )

    def test_ignores_compressed_entry_outside_archive_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["outside/unused.bin"] = b"unused"
            archive_path = self.write_fixture(
                temp_dir,
                entries,
                compressed_entries=("outside/unused.bin",),
            )
            result = run_helper("inspect", archive_path)

            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.rstrip(b"\r\n"), expected_layout())

    def test_rejects_missing_byteorder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            del entries["random-root/byteorder"]
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"random-root/byteorder is missing", result.stderr)

    def test_rejects_unknown_byteorder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = self.write_fixture(temp_dir, valid_entries(byteorder=b"middle"))
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"unsupported pt2 byteorder middle", result.stderr)

    def test_rejects_nested_model_json_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            del entries["random-root/models/model.json"]
            entries["random-root/models/nested/model.json"] = b"{}"
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"invalid ExportedProgram model entry random-root/models/nested/model.json", result.stderr)

    def test_rejects_model_entry_without_json_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            del entries["random-root/models/model.json"]
            entries["random-root/models/model.bin"] = b"{}"
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"invalid ExportedProgram model entry random-root/models/model.bin", result.stderr)

    def test_reads_json_and_blob_by_exact_entry_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["random-root/models/model.json"] = b'{"values":[1,true,null]}'
            archive_path = self.write_fixture(temp_dir, entries)

            json_result = run_helper("read-json", archive_path, "random-root/models/model.json")
            blob_result = run_helper("read-blob", archive_path, "random-root/models/model.json")

            self.assertEqual(json_result.returncode, 0, json_result.stderr.decode(errors="replace"))
            self.assertEqual(json_result.stdout.rstrip(b"\r\n"), b"object")
            self.assertEqual(blob_result.returncode, 0, blob_result.stderr.decode(errors="replace"))
            self.assertEqual(blob_result.stdout.rstrip(b"\r\n"), b"24")

    def test_json_error_names_entry_and_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["random-root/models/model.json"] = b'{"value":}'
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("read-json", archive_path, "random-root/models/model.json")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"invalid json random-root/models/model.json at line 1 column 10 byte 9", result.stderr)

    def test_missing_blob_names_exact_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = self.write_fixture(temp_dir)
            result = run_helper("read-blob", archive_path, "random-root/data/missing.bin")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"random-root/data/missing.bin is missing", result.stderr)

    def test_rejects_blob_with_corrupt_crc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = self.write_fixture(temp_dir)
            entry_name = "random-root/models/model.json"
            corrupt_entry_payload(archive_path, entry_name)
            result = run_helper("read-blob", archive_path, entry_name)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"cannot read random-root/models/model.json", result.stderr)

    def test_rejects_read_outside_discovered_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = valid_entries()
            entries["outside/data.bin"] = b"outside"
            archive_path = self.write_fixture(temp_dir, entries)
            result = run_helper("read-blob", archive_path, "outside/data.bin")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"pt2 entry is outside archive root outside/data.bin", result.stderr)

    def test_failed_reopen_clears_previous_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.pt2"
            invalid_path = Path(temp_dir) / "invalid.pt"
            write_archive(valid_path, valid_entries())
            write_archive(invalid_path, {"archive/data.pkl": b"payload"})

            result = run_helper("reopen", valid_path, str(invalid_path))

            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"model archive is not an exported program pt2 package", result.stdout)
            self.assertIn(b"pt2 package is not open", result.stdout)

    def test_rejects_non_pt2_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "torchscript-like.pt"
            write_archive(archive_path, {"archive/data.pkl": b"payload"})
            result = run_helper("inspect", archive_path)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"model archive is not an exported program pt2 package", result.stderr)

    def test_discovers_real_torch_export_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "real-tiny.pt2"
            exported_program = torch.export.export(TinyModule().eval(), (torch.ones(1),))
            torch.export.save(exported_program, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                marker = next(name for name in archive.namelist() if name.endswith("/archive_format"))
            root = marker[: -len("/archive_format")]

            result = run_helper("inspect", archive_path)

            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.rstrip(b"\r\n"), expected_layout(root=root))


if __name__ == "__main__":
    unittest.main()
