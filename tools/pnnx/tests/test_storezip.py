#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import io
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


STOREZIP_HELPER = Path(sys.argv[1])
sys.argv = [sys.argv[0]]


class NonSeekableBuffer(io.BytesIO):
    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation("seek")


def run_helper(*args):
    return subprocess.run(
        [str(STOREZIP_HELPER), *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_zip(path, entries, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def write_data_descriptor_zip(path, entries):
    stream = NonSeekableBuffer()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    path.write_bytes(stream.getvalue())


class StoreZipReaderTest(unittest.TestCase):
    def test_reads_stored_entry_with_data_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "model.pt2"
            entry_name = "arbitrary-root/archive_format"
            write_data_descriptor_zip(archive_path, {entry_name: b"pt2"})

            with zipfile.ZipFile(archive_path) as archive:
                self.assertNotEqual(archive.getinfo(entry_name).flag_bits & 0x08, 0)

            result = run_helper("read", archive_path, entry_name)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"pt2")

    def test_reopen_discards_entries_from_previous_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.zip"
            second_path = Path(temp_dir) / "second.zip"
            write_zip(first_path, {"old-entry": b"old"})
            write_zip(second_path, {"new-entry": b"new"})

            result = run_helper(
                "reopen-without-stale-entry",
                first_path,
                second_path,
                "old-entry",
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_reads_storezip_writer_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "writer.zip"
            result = run_helper("writer-round-trip", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_rejects_archive_without_eocd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "missing-eocd.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = archive_path.read_bytes()
            eocd_offset = data.rfind(b"PK\x05\x06")
            self.assertGreaterEqual(eocd_offset, 0)
            archive_path.write_bytes(data[:eocd_offset])

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_rejects_central_directory_outside_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "invalid-offset.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            eocd_offset = data.rfind(b"PK\x05\x06")
            self.assertGreaterEqual(eocd_offset, 0)
            struct.pack_into("<I", data, eocd_offset + 16, len(data) + 4096)
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_rejects_compressed_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "deflated.zip"
            write_zip(archive_path, {"payload": b"data"}, zipfile.ZIP_DEFLATED)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
