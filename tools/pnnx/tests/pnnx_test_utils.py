# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
from enum import Enum
from pathlib import Path
import re
import subprocess
import sys

import torch

from pt2_expectations import NUMERICAL_MISMATCH
from pt2_expectations import PASS
from pt2_expectations import PT2_EXPECTATIONS
from pt2_expectations import PT2_FRONTEND_UNSUPPORTED
from pt2_expectations import PNNX_LOWERING_UNSUPPORTED
from pt2_expectations import UNCLASSIFIED
from pt2_expectations import EXPORT_UNSUPPORTED


SUPPORTED = "SUPPORTED"
LEGACY_PT2_UNSUPPORTED = "LEGACY_PT2_UNSUPPORTED"
PT2_PRODUCER_VERSION_UNTESTED = "PT2_PRODUCER_VERSION_UNTESTED"


class ExportTestFormat(Enum):
    TORCHSCRIPT = "torchscript"
    EXPORTED_PROGRAM = "pt2"


def _torch_version_tuple():
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch.__version__)
    if match is None:
        return None

    return tuple(int(value or 0) for value in match.groups())


def pt2_producer_status():
    version = _torch_version_tuple()
    if version is None or version > (2, 12, 1):
        return PT2_PRODUCER_VERSION_UNTESTED
    if version < (2, 9, 0):
        return LEGACY_PT2_UNSUPPORTED
    return SUPPORTED


def _selected_format():
    value = os.environ.get("PNNX_TEST_FORMAT", ExportTestFormat.TORCHSCRIPT.value)
    try:
        return ExportTestFormat(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ExportTestFormat)
        raise ValueError("invalid PNNX_TEST_FORMAT '%s', expected one of: %s" % (value, choices)) from exc


def _expectation(basename):
    if basename not in PT2_EXPECTATIONS:
        raise AssertionError("missing pt2 expectation for %s" % basename)
    return PT2_EXPECTATIONS[basename]


def _handle_pt2_failure(basename, category, detail):
    expected_category, expected_substring = _expectation(basename)

    if expected_category == UNCLASSIFIED:
        raise AssertionError(
            "%s produced an unclassified pt2 failure\n"
            "category: %s\n"
            "diagnostic:\n%s" % (basename, category, detail)
        )

    if expected_category != category:
        raise AssertionError(
            "%s pt2 failure category changed: expected %s, got %s\n%s"
            % (basename, expected_category, category, detail)
        )

    if not expected_substring:
        raise AssertionError("%s expected failure has an empty diagnostic substring" % basename)

    if expected_substring not in detail:
        raise AssertionError(
            "%s pt2 diagnostic changed: expected substring %r\n%s"
            % (basename, expected_substring, detail)
        )

    print("%s: expected pt2 failure %s: %s" % (basename, category, expected_substring))
    return None


def _handle_pt2_conversion_success(basename):
    expected_category, _ = _expectation(basename)
    if expected_category not in (PASS, NUMERICAL_MISMATCH):
        raise AssertionError(
            "%s pt2 conversion now passes; update its expectation from %s to PASS"
            % (basename, expected_category)
        )


def _remove_if_present(path):
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _import_generated_module(path, basename):
    module_name = "_pnnx_test_%s" % re.sub(r"[^0-9A-Za-z_]", "_", basename)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create module spec for %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def convert_and_import(net, inputs, basename, pnnx_args=(), trace_kwargs=None):
    if not isinstance(inputs, tuple):
        raise TypeError("inputs must be a tuple")
    if not isinstance(pnnx_args, tuple):
        raise TypeError("pnnx_args must be a tuple")
    if trace_kwargs is None:
        trace_kwargs = {}
    if not isinstance(trace_kwargs, dict):
        raise TypeError("trace_kwargs must be a dict")

    export_format = _selected_format()
    if export_format == ExportTestFormat.EXPORTED_PROGRAM:
        producer_status = pt2_producer_status()
        if producer_status != SUPPORTED:
            print("%s: pt2 producer gate: %s (torch %s)" % (basename, producer_status, torch.__version__))
            return None

    output_basename = basename
    if export_format == ExportTestFormat.EXPORTED_PROGRAM:
        output_basename += "_pt2"

    archive_suffix = ".pt2" if export_format == ExportTestFormat.EXPORTED_PROGRAM else ".pt"
    archive_path = Path(output_basename + archive_suffix)
    generated_path = Path(output_basename + "_pnnx.py")
    _remove_if_present(archive_path)
    _remove_if_present(generated_path)

    try:
        if export_format == ExportTestFormat.EXPORTED_PROGRAM:
            exported_program = torch.export.export(net, inputs)
            torch.export.save(exported_program, archive_path)
        else:
            traced_module = torch.jit.trace(net, inputs, **trace_kwargs)
            traced_module.save(str(archive_path))
    except Exception as exc:
        if export_format == ExportTestFormat.EXPORTED_PROGRAM:
            detail = "%s: %s" % (type(exc).__name__, exc)
            return _handle_pt2_failure(basename, EXPORT_UNSUPPORTED, detail)
        raise

    pnnx_executable = os.environ.get("PNNX_TEST_PNNX", "../src/pnnx")
    command = [pnnx_executable, str(archive_path)] + list(pnnx_args)
    completed = subprocess.run(command, capture_output=True, text=True)
    diagnostic = completed.stdout + completed.stderr
    if completed.returncode != 0:
        if export_format == ExportTestFormat.EXPORTED_PROGRAM:
            category = PNNX_LOWERING_UNSUPPORTED
            if "load exported program failed:" in diagnostic:
                category = PT2_FRONTEND_UNSUPPORTED
            return _handle_pt2_failure(basename, category, diagnostic)
        raise RuntimeError("pnnx failed with exit code %d\n%s" % (completed.returncode, diagnostic))

    try:
        module = _import_generated_module(generated_path, output_basename)
    except Exception as exc:
        if export_format == ExportTestFormat.EXPORTED_PROGRAM:
            detail = "%s: %s" % (type(exc).__name__, exc)
            return _handle_pt2_failure(basename, PNNX_LOWERING_UNSUPPORTED, detail)
        raise

    if export_format == ExportTestFormat.EXPORTED_PROGRAM:
        _handle_pt2_conversion_success(basename)
    return module


def check_numerical_result(basename, passed):
    if _selected_format() != ExportTestFormat.EXPORTED_PROGRAM:
        return passed

    if passed:
        expected_category, _ = _expectation(basename)
        if expected_category != PASS:
            raise AssertionError(
                "%s pt2 numerical comparison now passes; update its expectation from %s to PASS"
                % (basename, expected_category)
            )
        return True

    result = _handle_pt2_failure(basename, NUMERICAL_MISMATCH, "output mismatch")
    return result is None
