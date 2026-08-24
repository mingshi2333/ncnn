#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from unittest import mock

import torch

from pnnx_test_utils import LEGACY_PT2_UNSUPPORTED
from pnnx_test_utils import PT2_PRODUCER_VERSION_UNTESTED
from pnnx_test_utils import SUPPORTED
from pnnx_test_utils import pt2_producer_status


class Pt2ProducerStatusTest(unittest.TestCase):
    def assert_status(self, torch_version, expected):
        with mock.patch.object(torch, "__version__", torch_version):
            self.assertEqual(pt2_producer_status(), expected)

    def test_supported_raw_payload_producers(self):
        self.assert_status("2.9.0", SUPPORTED)
        self.assert_status("2.12.1+cu126", SUPPORTED)
        self.assert_status("2.13.0+cpu", SUPPORTED)

    def test_unsupported_producers(self):
        self.assert_status("2.8.0", LEGACY_PT2_UNSUPPORTED)
        self.assert_status("2.12.2", PT2_PRODUCER_VERSION_UNTESTED)
        self.assert_status("2.13.1", PT2_PRODUCER_VERSION_UNTESTED)
        self.assert_status("2.14.0", PT2_PRODUCER_VERSION_UNTESTED)


if __name__ == "__main__":
    unittest.main()
