#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import io
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
import zlib
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


def write_descriptor_variant_zip(path, name, payload, with_signature, zip64):
    name_bytes = name.encode("utf-8")
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    flag = 0x0808
    version = 45 if zip64 else 20

    if zip64:
        local_compressed_size = 0xFFFFFFFF
        local_uncompressed_size = 0xFFFFFFFF
        local_extra = struct.pack("<HHQQ", 0x0001, 16, len(payload), len(payload))
        central_compressed_size = 0xFFFFFFFF
        central_uncompressed_size = 0xFFFFFFFF
        central_extra = struct.pack("<HHQQ", 0x0001, 16, len(payload), len(payload))
    else:
        local_compressed_size = 0
        local_uncompressed_size = 0
        local_extra = b""
        central_compressed_size = len(payload)
        central_uncompressed_size = len(payload)
        central_extra = b""

    local_header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        version,
        flag,
        0,
        0,
        0,
        0,
        local_compressed_size,
        local_uncompressed_size,
        len(name_bytes),
        len(local_extra),
    )

    descriptor = b""
    if with_signature:
        descriptor += struct.pack("<I", 0x08074B50)
    if zip64:
        descriptor += struct.pack("<IQQ", crc32, len(payload), len(payload))
    else:
        descriptor += struct.pack("<III", crc32, len(payload), len(payload))

    central_offset = len(local_header) + len(name_bytes) + len(local_extra) + len(payload) + len(descriptor)
    central_header = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        version,
        version,
        flag,
        0,
        0,
        0,
        crc32,
        central_compressed_size,
        central_uncompressed_size,
        len(name_bytes),
        len(central_extra),
        0,
        0,
        0,
        0,
        0,
    )
    central_directory = central_header + name_bytes + central_extra
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        1,
        1,
        len(central_directory),
        central_offset,
        0,
    )

    path.write_bytes(
        local_header
        + name_bytes
        + local_extra
        + payload
        + descriptor
        + central_directory
        + eocd
    )


def local_header_offset(data):
    offset = data.find(b"PK\x03\x04")
    if offset < 0:
        raise AssertionError("local file header not found")
    return offset


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

    def test_reads_descriptor_without_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "descriptor-no-signature.zip"
            write_descriptor_variant_zip(
                archive_path,
                "payload",
                b"descriptor-without-signature",
                with_signature=False,
                zip64=False,
            )

            result = run_helper("read", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"descriptor-without-signature")

    def test_reads_descriptor_with_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "descriptor-signature.zip"
            write_descriptor_variant_zip(
                archive_path,
                "payload",
                b"descriptor-with-signature",
                with_signature=True,
                zip64=False,
            )

            result = run_helper("read", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"descriptor-with-signature")

    def test_reads_zip64_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "descriptor-zip64.zip"
            write_descriptor_variant_zip(
                archive_path,
                "payload",
                b"zip64-descriptor",
                with_signature=True,
                zip64=True,
            )

            result = run_helper("read", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"zip64-descriptor")

    def test_reads_zip64_local_size_sentinel_without_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "force-zip64.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                with archive.open("payload", "w", force_zip64=True) as entry:
                    entry.write(b"force-zip64")

            result = run_helper("read", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"force-zip64")

    def test_reads_archive_with_eocd_comment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "comment.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("payload", b"commented")
                archive.comment = b"pnnx-test-comment"

            result = run_helper("read", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"commented")

    def test_reads_empty_stored_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "empty.zip"
            write_zip(archive_path, {"empty": b""})

            result = run_helper("read", archive_path, "empty")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout, b"")

    def test_rejects_duplicate_entry_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("payload", b"first")
                    archive.writestr("payload", b"second")

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"duplicate zip entry name payload", result.stderr)

    def test_rejects_local_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "local-name.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            name_offset = offset + 30
            data[name_offset] ^= 0x01
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip local file name mismatch payload", result.stderr)

    def test_rejects_local_flag_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "local-flag.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            flag = struct.unpack_from("<H", data, offset + 6)[0]
            struct.pack_into("<H", data, offset + 6, flag ^ 0x0800)
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip local file flag mismatch payload", result.stderr)

    def test_rejects_local_method_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "local-method.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            struct.pack_into("<H", data, offset + 8, 8)
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip local file compression mismatch payload", result.stderr)

    def test_rejects_local_crc_mismatch_without_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "local-crc.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            crc32 = struct.unpack_from("<I", data, offset + 14)[0]
            struct.pack_into("<I", data, offset + 14, crc32 ^ 0xFFFFFFFF)
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip local file crc mismatch payload", result.stderr)

    def test_rejects_local_size_mismatch_without_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "local-size.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            compressed_size = struct.unpack_from("<I", data, offset + 18)[0]
            struct.pack_into("<I", data, offset + 18, compressed_size + 1)
            archive_path.write_bytes(data)

            result = run_helper("expect-open-failure", archive_path)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip local file size mismatch payload", result.stderr)

    def test_rejects_crc_mismatch_on_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "crc.zip"
            write_zip(archive_path, {"payload": b"data"})
            data = bytearray(archive_path.read_bytes())
            offset = local_header_offset(data)
            name_length = struct.unpack_from("<H", data, offset + 26)[0]
            extra_length = struct.unpack_from("<H", data, offset + 28)[0]
            payload_offset = offset + 30 + name_length + extra_length
            data[payload_offset] ^= 0x01
            archive_path.write_bytes(data)

            result = run_helper(
                "expect-read-failure-after-open",
                archive_path,
                "payload",
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"zip entry crc mismatch payload", result.stderr)

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

    def test_indexes_but_does_not_read_deflate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "deflated.zip"
            write_zip(archive_path, {"payload": b"data"}, zipfile.ZIP_DEFLATED)

            result = run_helper("expect-compressed-entry", archive_path, "payload")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

            result = run_helper(
                "expect-read-failure-after-open",
                archive_path,
                "payload",
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertIn(b"compressed zip entry is not supported payload method=8", result.stderr)


if __name__ == "__main__":
    unittest.main()
