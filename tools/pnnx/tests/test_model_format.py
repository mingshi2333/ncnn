#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import torch


MODEL_FORMAT_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


def run_helper(path):
    return subprocess.run(
        [str(MODEL_FORMAT_HELPER), "detect", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_zip(path, entries, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def write_pt2_marker(path, root="archive", version=b"0"):
    write_zip(
        path,
        [
            (f"{root}/archive_format", b"pt2"),
            (f"{root}/archive_version", version),
        ],
    )


class TinyModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class ModelFormatTest(unittest.TestCase):
    def test_pt2_renamed_to_bin_is_pt2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "renamed.bin"
            write_pt2_marker(archive_path, root="arbitrary-root")

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"pt2|arbitrary-root|0"])

    def test_torchscript_renamed_to_pt2_is_torchscript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "renamed.pt2"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                traced = torch.jit.trace(TinyModule().eval(), (torch.ones(1),))
                torch.jit.save(traced, str(archive_path))

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"torchscript"])

    def test_non_zip_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            model_path.write_bytes(b"not-a-zip")

            result = run_helper(model_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"unknown"])

    def test_malformed_zip_candidate_is_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "broken.zip"
            archive_path.write_bytes(b"PK\x03\x04broken")

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"invalid zip model archive", result.stderr)

    def test_empty_zip_is_torchscript_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "empty.zip"
            with zipfile.ZipFile(archive_path, "w"):
                pass

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"torchscript"])

    def test_zip_without_marker_is_torchscript_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "generic.zip"
            write_zip(archive_path, [("archive/data.pkl", b"payload")])

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"torchscript"])

    def test_requires_unique_archive_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "two-roots.pt2"
            write_zip(
                archive_path,
                [
                    ("first/archive_format", b"pt2"),
                    ("first/archive_version", b"0"),
                    ("second/archive_format", b"pt2"),
                    ("second/archive_version", b"0"),
                ],
            )

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"multiple archive_format entries", result.stderr)

    def test_requires_matching_archive_version_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "split-root.pt2"
            write_zip(
                archive_path,
                [
                    ("model/archive_format", b"pt2"),
                    ("other/archive_version", b"0"),
                ],
            )

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"model/archive_version is missing", result.stderr)

    def test_rejects_non_decimal_archive_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "hex-version.pt2"
            write_pt2_marker(archive_path, version=b"0x0")

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"archive_version is not an unsigned decimal integer", result.stderr)

    def test_rejects_archive_version_overflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "overflow-version.pt2"
            write_pt2_marker(archive_path, version=b"18446744073709551616")

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"archive_version is out of uint64 range", result.stderr)

    def test_accepts_maximum_uint64_archive_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "max-version.pt2"
            write_pt2_marker(archive_path, version=b"18446744073709551615")

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(
                result.stdout.splitlines(),
                [b"pt2|archive|18446744073709551615"],
            )

    def test_accepts_archive_version_leading_zeroes_within_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "leading-zeroes.pt2"
            write_pt2_marker(archive_path, version=b"00000000000000000001")

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"pt2|archive|1"])

    def test_rejects_archive_version_payload_longer_than_uint64_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "long-version.pt2"
            write_pt2_marker(archive_path, version=b"0" * 21)

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"archive_version payload is longer than 20 bytes", result.stderr)

    def test_rejects_missing_archive_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "missing-version.pt2"
            write_zip(archive_path, [("model/archive_format", b"pt2")])

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"model/archive_version is missing", result.stderr)

    def test_rejects_unknown_archive_format_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "unknown-format.pt2"
            write_zip(
                archive_path,
                [
                    ("model/archive_format", b"pt3"),
                    ("model/archive_version", b"0"),
                ],
            )

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"archive_format payload is not pt2", result.stderr)

    def test_rejects_compressed_archive_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "compressed-marker.pt2"
            write_zip(
                archive_path,
                [
                    ("model/archive_format", b"pt2"),
                    ("model/archive_version", b"0"),
                ],
                compression=zipfile.ZIP_DEFLATED,
            )

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"model/archive_format must use zip store compression", result.stderr)

    def test_rejects_compressed_archive_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "compressed-version.pt2"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("model/archive_format", b"pt2", compress_type=zipfile.ZIP_STORED)
                archive.writestr("model/archive_version", b"0", compress_type=zipfile.ZIP_DEFLATED)

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"model/archive_version must use zip store compression", result.stderr)

    def test_rejects_corrupt_archive_format_crc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "corrupt-marker.pt2"
            write_pt2_marker(archive_path, root="model")
            data = bytearray(archive_path.read_bytes())
            local_header_offset = data.find(b"PK\x03\x04")
            self.assertGreaterEqual(local_header_offset, 0)
            name_length = int.from_bytes(data[local_header_offset + 26 : local_header_offset + 28], "little")
            extra_length = int.from_bytes(data[local_header_offset + 28 : local_header_offset + 30], "little")
            payload_offset = local_header_offset + 30 + name_length + extra_length
            data[payload_offset] ^= 0x01
            archive_path.write_bytes(data)

            result = run_helper(archive_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"cannot read model/archive_format", result.stderr)

    def test_classifies_aoti_only_package_as_pt2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "aoti.pt2"
            write_zip(
                archive_path,
                [
                    ("package/archive_format", b"pt2"),
                    ("package/archive_version", b"0"),
                    ("package/aotinductor/model.so", b"native-code"),
                ],
            )

            result = run_helper(archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.splitlines(), [b"pt2|package|0"])


if __name__ == "__main__":
    unittest.main()
