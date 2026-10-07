#!/usr/bin/env python3
"""Fork-only adapter for the checked-out upstream pnnx workflow.

Reuse its dependency builds, version matrix, and test commands on hosted
runners. Do not modify expectations, tolerances, or the upstream workflow.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/pnnx.yml').read_text())
FIX = '6077e429c6b3e9f1dd4610b4dc3922db631444b7'
BASE = '7af5e93d10df7cdc05a776a92e3559fed52b3cfa'
CI_FILES = {'.github/scripts/pnnx_fork_ci.py', '.github/workflows/pnnx-fork-validation.yml'}


def output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        stream.write(name + '=' + json.dumps(value, separators=(',', ':')) + '\n')


def identity():
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if sha != os.environ['VALIDATION_SHA']:
        raise RuntimeError('checkout does not match the requested validation commit')
    print('checkout:', sha, 'python:', sys.executable, sys.version, flush=True)
    for name in ('tools/pnnx/tests/pnnx_test_utils.py', 'tools/pnnx/tests/test_pnnx_test_utils.py',
                 'tools/pnnx/src/load_exported_program.cpp'):
        data = (ROOT / name).read_bytes()
        print(name, 'lines=', len(data.splitlines()), 'sha256=', hashlib.sha256(data).hexdigest(), flush=True)


def metadata():
    identity()
    subprocess.run(['git', 'merge-base', '--is-ancestor', FIX, 'HEAD'], cwd=ROOT, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', FIX, 'HEAD'], cwd=ROOT, text=True)
    if not set(changed.splitlines()).issubset(CI_FILES):
        raise RuntimeError('validation branch contains code not included in the promotion commit')
    subprocess.run(['git', 'diff', '--check', BASE, 'HEAD'], cwd=ROOT, check=True)
    matrix = WORKFLOW['jobs']['test']['strategy']['matrix']
    versions = [str(row['torch']) for row in matrix['include']]
    if len(versions) != 19 or '2.12.1' not in versions:
        raise RuntimeError('upstream matrix changed; review the validation gate')
    output('matrix', matrix)
    output('env', {key: str(value) for key, value in WORKFLOW['env'].items()})
    print('Full upstream matrix:', versions, flush=True)


def render(command, env, matrix):
    def substitute(match):
        field = match.group(1).strip()
        if field == 'github.workspace':
            return ROOT.as_posix()
        if field.startswith('env.'):
            return env[field[4:]]
        if field.startswith('matrix.'):
            return str(matrix[field[7:]])
        raise ValueError('unsupported workflow expression: ' + field)
    return re.sub(r'\$\{\{\s*(.*?)\s*\}\}', substitute, command)


def check_report(path, required=()):
    cases = ET.parse(path).getroot().findall('.//testcase')
    if not cases:
        raise RuntimeError('no test results: ' + str(path))
    passed = {case.get('name') for case in cases
              if case.find('failure') is None and case.find('error') is None
              and case.find('skipped') is None and case.get('status') != 'notrun'}
    if not set(required).issubset(passed):
        raise RuntimeError('mandatory frontend/native cases missing or skipped: ' + repr(sorted(set(required) - passed)))
    print('Checked report:', path.name, 'total=', len(cases), 'non-skipped=', len(passed), flush=True)


def run_steps(job, names):
    identity()
    matrix = json.loads(os.environ.get('MATRIX_JSON', '{}'))
    env = os.environ.copy()
    env.update({key: str(value) for key, value in WORKFLOW['env'].items()})
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
    env['EXTRA_CMAKE_ARGS'] = '-DNCNN_VULKAN=OFF'
    for key, value in WORKFLOW['jobs'][job].get('env', {}).items():
        env[key] = render(str(value), env, matrix)
    if 'PYTHONUSERBASE' in env:
        env['PATH'] = str(Path(env['PYTHONUSERBASE']) / 'bin') + os.pathsep + env['PATH']
        # Plain Python steps (DLL setup and helper tests) need the same user site.
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write('PYTHONUSERBASE=' + env['PYTHONUSERBASE'] + '\n')
    reports = ROOT / 'fork-ci-reports'
    reports.mkdir(exist_ok=True)
    for name in names:
        matches = [step for step in WORKFLOW['jobs'][job]['steps'] if step.get('name') == name]
        if len(matches) != 1 or 'run' not in matches[0]:
            raise ValueError('cannot resolve upstream run step: ' + job + '/' + name)
        step = matches[0]
        step_env = env.copy()
        for key, value in step.get('env', {}).items():
            step_env[key] = render(str(value), step_env, matrix)
        step_env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
        command = render(step['run'], step_env, matrix)
        # Hosted runners have less RAM than the upstream self-hosted builder.
        command = re.sub(r'-j\s+[48]\b', '-j 2', command)
        command = command.replace('CMAKE_BUILD_PARALLEL_LEVEL=8', 'CMAKE_BUILD_PARALLEL_LEVEL=2')
        report = reports / (job + '-' + name + '.xml')
        has_ctest = re.search(r'\bctest\s', command) is not None
        if has_ctest:
            command = re.sub(r'\bctest\s', 'ctest --no-tests=error --output-junit "' + report.as_posix() + '" ', command)
        required = set()
        if name == 'exported-program-test':
            inventory = json.loads(subprocess.check_output(
                ['ctest', '-C', 'Release', '--show-only=json-v1'],
                cwd=ROOT / 'tools/pnnx/build', env=step_env, text=True))
            for test in inventory['tests']:
                labels = next((prop['value'] for prop in test.get('properties', [])
                               if prop['name'] == 'LABELS'), [])
                if set(labels) & {'pt2_frontend', 'pt2_ncnn'}:
                    required.add(test['name'])
            if len(required) < 12:
                raise RuntimeError('required frontend/native test registration is incomplete')
        print('::group::' + job + '/' + name, flush=True)
        print(command, flush=True)
        subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', command], cwd=ROOT, env=step_env, check=True)
        if has_ctest:
            check_report(report, required=required)
        print('::endgroup::', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['metadata']:
        metadata()
    elif sys.argv[1:] == ['identity']:
        identity()
    elif len(sys.argv) >= 4 and sys.argv[1] == 'run':
        run_steps(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit('usage: pnnx_fork_ci.py metadata|identity|run JOB STEP...')
