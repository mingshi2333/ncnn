#!/usr/bin/env python3
"""Audit fork-only adaptations against the immutable clean source candidate."""
import copy
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = '66a109e4b0ce92be3cfe8b10b514510e3b89f84f'
CI_FILES = set([
    ".github/scripts/ncnn_fork_ci.py",
    ".github/scripts/ncnn_fork_setup.sh",
    ".github/scripts/pnnx_fork_ci.py",
    ".github/workflows/android.yml",
    ".github/workflows/code-format.yml",
    ".github/workflows/codeql-analysis.yml",
    ".github/workflows/compare-binary-size.yml",
    ".github/workflows/elf-riscv32.yml",
    ".github/workflows/elf-riscv64.yml",
    ".github/workflows/esp32.yml",
    ".github/workflows/harmonyos.yml",
    ".github/workflows/ios.yml",
    ".github/workflows/linux-aarch64.yml",
    ".github/workflows/linux-arm.yml",
    ".github/workflows/linux-loongarch64.yml",
    ".github/workflows/linux-mips.yml",
    ".github/workflows/linux-mips64.yml",
    ".github/workflows/linux-ppc64.yml",
    ".github/workflows/linux-riscv32.yml",
    ".github/workflows/linux-riscv64.yml",
    ".github/workflows/linux-x64-cpu-clang.yml",
    ".github/workflows/linux-x64-cpu-gcc-musl.yml",
    ".github/workflows/linux-x64-cpu-gcc.yml",
    ".github/workflows/linux-x64-gpu-clang.yml",
    ".github/workflows/linux-x64-gpu-gcc.yml",
    ".github/workflows/linux-x64-sde.yml",
    ".github/workflows/linux-x86-cpu-clang.yml",
    ".github/workflows/linux-x86-cpu-gcc.yml",
    ".github/workflows/mac-catalyst.yml",
    ".github/workflows/macos.yml",
    ".github/workflows/ncnn-fork-validation.yml",
    ".github/workflows/pnnx-fork-validation.yml",
    ".github/workflows/python.yml",
    ".github/workflows/test-coverage.yml",
    ".github/workflows/tvos.yml",
    ".github/workflows/visionos.yml",
    ".github/workflows/watchos.yml",
    ".github/workflows/web-assembly.yml",
    ".github/workflows/windows-arm.yml",
    ".github/workflows/windows-clang.yml",
    ".github/workflows/windows-mingw.yml",
    ".github/workflows/windows-xp.yml",
    ".github/workflows/windows.yml"
])
WORKFLOWS = [
    "android.yml",
    "code-format.yml",
    "codeql-analysis.yml",
    "compare-binary-size.yml",
    "elf-riscv32.yml",
    "elf-riscv64.yml",
    "esp32.yml",
    "harmonyos.yml",
    "ios.yml",
    "linux-aarch64.yml",
    "linux-arm.yml",
    "linux-loongarch64.yml",
    "linux-mips.yml",
    "linux-mips64.yml",
    "linux-ppc64.yml",
    "linux-riscv32.yml",
    "linux-riscv64.yml",
    "linux-x64-cpu-clang.yml",
    "linux-x64-cpu-gcc-musl.yml",
    "linux-x64-cpu-gcc.yml",
    "linux-x64-gpu-clang.yml",
    "linux-x64-gpu-gcc.yml",
    "linux-x64-sde.yml",
    "linux-x86-cpu-clang.yml",
    "linux-x86-cpu-gcc.yml",
    "mac-catalyst.yml",
    "macos.yml",
    "python.yml",
    "test-coverage.yml",
    "tvos.yml",
    "visionos.yml",
    "watchos.yml",
    "web-assembly.yml",
    "windows-arm.yml",
    "windows-clang.yml",
    "windows-mingw.yml",
    "windows-xp.yml",
    "windows.yml"
]


def read_workflow(text):
    # Keep ON/OFF as strings, as GitHub's YAML 1.2 parser does.
    return yaml.load(text, Loader=yaml.BaseLoader)


def steps_for_comparison(job, filename):
    result = []
    source_checkout = True
    for value in job['steps']:
        step = copy.deepcopy(value)
        if step.get('name', '').startswith('fork-'):
            continue
        if filename == 'codeql-analysis.yml' and step.get('run') == 'git checkout HEAD^2':
            continue
        if step.get('name') == 'codecov':
            # Preserve coverage collection; replace only the external uploader.
            if step.get('uses') == 'actions/upload-artifact@v7':
                assert step['with']['path'] == 'build*/lcov.info'
                assert step['with']['if-no-files-found'] == 'error'
            else:
                assert step['uses'] == 'codecov/codecov-action@v6'
            continue
        if step.get('uses', '').startswith('actions/checkout@') and source_checkout:
            source_checkout = False
            params = step.setdefault('with', {})
            for key in ('ref', 'persist-credentials'):
                params.pop(key, None)
            if not params:
                step.pop('with')
        elif filename == 'compare-binary-size.yml' and step.get('name') == 'checkout-base-branch':
            step['with'].pop('ref')
        if 'run' in step:
            step['run'] = step['run'].rstrip()
            step['run'] = step['run'].replace('ctest --no-tests=error ', 'ctest ')
            step['run'] = re.sub(r'-j [248]\b', '-j PARALLEL', step['run'])
        result.append(step)
    return result


def validate_workflow(original, current, filename):
    if set(original['jobs']) != set(current['jobs']):
        raise ValueError(filename + ': original jobs were added or removed')
    if original.get('env') != current.get('env'):
        raise ValueError(filename + ': original workflow environment changed')
    for name, old in original['jobs'].items():
        new = current['jobs'][name]
        for key in ('env', 'needs'):
            if old.get(key) != new.get(key):
                raise ValueError(filename + '/' + name + ': changed ' + key)
        if old.get('strategy', {}).get('matrix') != new.get('strategy', {}).get('matrix'):
            raise ValueError(filename + '/' + name + ': original matrix changed')
        if steps_for_comparison(old, filename) != steps_for_comparison(new, filename):
            raise ValueError(filename + '/' + name + ': original build/test steps changed')


def matrix_rows(job):
    matrix = job.get('strategy', {}).get('matrix')
    if not matrix:
        return [{}]
    axes = {k: v for k, v in matrix.items() if k not in ('include', 'exclude')}
    if not axes:
        return matrix.get('include', [])
    if 'include' in matrix or 'exclude' in matrix:
        raise ValueError('Review mixed matrix expansion before changing the inventory')
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def verify():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != os.environ.get('VALIDATION_SHA', head):
        raise ValueError('Wrong validation checkout')
    subprocess.run(['git', 'merge-base', '--is-ancestor', CANDIDATE, 'HEAD'], cwd=ROOT, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', CANDIDATE], cwd=ROOT, text=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True)
    if set(changed.splitlines() + untracked.splitlines()) != CI_FILES:
        raise ValueError('Candidate/validation difference does not match exact fork-only CI allowlist')
    inventory = []
    for filename in WORKFLOWS:
        path = '.github/workflows/' + filename
        original = read_workflow(subprocess.check_output(['git', 'show', CANDIDATE + ':' + path], cwd=ROOT, text=True))
        current = read_workflow((ROOT / path).read_text())
        if filename != 'code-format.yml':
            validate_workflow(original, current, filename)
            if 'pnnx-exported-program' not in current['on']['pull_request']['branches']:
                raise ValueError(path + ': validation PR cannot trigger workflow')
            if 'paths' in current['on']['pull_request'] and '.github/scripts/ncnn_fork_ci.py' not in current['on']['pull_request']['paths']:
                raise ValueError(path + ': missing common full-suite trigger')
        else:
            assert current['permissions']['contents'] == 'read'
            assert any(s.get('run') == 'git diff --exit-code' for s in current['jobs']['code-format']['steps'])
            assert not any('git-auto-commit' in s.get('uses', '') for s in current['jobs']['code-format']['steps'])
        for job_id, job in current['jobs'].items():
            for row in matrix_rows(job):
                inventory.append({'workflow': filename, 'job': job_id, 'matrix': row, 'runner': job['runs-on']})
    result = {'candidate': CANDIDATE, 'validation_head': head, 'workflow_count': len(WORKFLOWS),
              'job_count': len(inventory), 'jobs': inventory,
              'also_required': 'pnnx-fork-validation: all 26 jobs including the full 19-version matrix; separate CodeQL, formatting and review statuses',
              'hardware_gap': 'Two T4 Vulkan jobs require a real NVIDIA T4 runner. They remain required and cannot be reported as passed.',
              'sdk_gap': 'Five Xuantie jobs require exact vendor GCC V3.4.0 and QEMU V5.4.1 packages and checksums; no version substitution.'}
    print(json.dumps(result, indent=2))
    return result


def self_test():
    old = {'jobs': {'test': {'steps': [{'run': 'cmake -DNCNN_VULKAN=ON ..\nctest --output-on-failure -j 8'}],
                                  'strategy': {'matrix': {'os': ['linux', 'windows']}}}}}
    new = copy.deepcopy(old)
    new['jobs']['test']['steps'][0]['run'] = 'cmake -DNCNN_VULKAN=ON ..\nctest --no-tests=error --output-on-failure -j 2'
    validate_workflow(old, new, 'probe')
    probes = []
    bad = copy.deepcopy(new)
    bad['jobs'].pop('test')
    probes.append(bad)
    bad = copy.deepcopy(new)
    bad['jobs']['test']['strategy']['matrix']['os'].pop()
    probes.append(bad)
    bad = copy.deepcopy(new)
    bad['jobs']['test']['steps'][0]['run'] = 'cmake -DNCNN_VULKAN=OFF ..\nctest --output-on-failure -j 2'
    probes.append(bad)
    bad = copy.deepcopy(new)
    bad['jobs']['test']['steps'].clear()
    probes.append(bad)
    for bad in probes:
        try:
            validate_workflow(old, bad, 'probe')
        except ValueError:
            continue
        raise AssertionError('audit accepted a weakened workflow')
    print('Audit regressions passed: allowed provisioning, removed job, removed matrix, changed option, removed test.')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('verify', 'self-test'):
        raise SystemExit('usage: ncnn_fork_ci.py verify|self-test')
    if sys.argv[1] == 'verify':
        verify()
    else:
        self_test()
