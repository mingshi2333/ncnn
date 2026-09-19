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
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[2]
FIX = '66a109e4b0ce92be3cfe8b10b514510e3b89f84f'
BASE = '7af5e93d10df7cdc05a776a92e3559fed52b3cfa'
CI_FILES = {'.github/scripts/pnnx_fork_ci.py', '.github/workflows/pnnx-fork-validation.yml'}
CODE_FILES = set(['tools/pnnx/src/ir.cpp', 'tools/pnnx/src/save_onnx.cpp', 'tools/pnnx/tests/CMakeLists.txt', 'tools/pnnx/tests/ncnn/test_attribute.py', 'tools/pnnx/tests/pnnx_test_utils.py', 'tools/pnnx/tests/pt2_expectations.py', 'tools/pnnx/tests/test_exported_program_parameter_roundtrip.cpp', 'tools/pnnx/tests/test_pnnx_test_utils.py', 'tools/pnnx/tests/test_python_codegen_paths.py'])
VERSIONS = ['1.8.1', '1.9.1', '1.10.0', '1.11.0', '1.12.0', '1.13.0',
            '2.0.0', '2.1.0', '2.2.1', '2.3.0', '2.4.0', '2.5.0', '2.6.0',
            '2.7.0', '2.8.0', '2.9.0', '2.10.0', '2.11.0', '2.12.1']
REQUIRED_PT2_TESTS = {
    'test_exported_program_parameter_roundtrip',
    'test_exported_program',
    'test_exported_program_roundtrip',
    'test_pnnx_test_utils',
    'test_python_codegen_paths',
    'test_pt2_ncnn_attribute',
    'test_pt2_ncnn_output',
    'test_pt2_ncnn_nn_BatchNorm2d',
    'test_pt2_ncnn_nn_Conv2d',
    'test_pt2_ncnn_nn_Embedding',
    'test_pt2_ncnn_nn_LayerNorm',
    'test_pt2_ncnn_nn_Linear',
    'test_pt2_ncnn_resnet18',
}


def upstream_workflow():
    return yaml.safe_load((ROOT / '.github/workflows/pnnx.yml').read_text(encoding='utf-8'))


def output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        stream.write(name + '=' + json.dumps(value, separators=(',', ':')) + '\n')


def identity():
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if sha != os.environ['VALIDATION_SHA']:
        raise RuntimeError('checkout does not match the requested validation commit')
    print('checkout:', sha, 'python:', sys.executable, sys.version, flush=True)
    for name in ('tools/pnnx/tests/pnnx_test_utils.py', 'tools/pnnx/tests/test_pnnx_test_utils.py',
                 'tools/pnnx/tests/ncnn/test_attribute.py', 'tools/pnnx/src/load_exported_program.cpp'):
        data = (ROOT / name).read_bytes()
        print(name, 'lines=', len(data.splitlines()), 'sha256=', hashlib.sha256(data).hexdigest(), flush=True)


def metadata():
    identity()
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, FIX], cwd=ROOT, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, FIX], cwd=ROOT, text=True)
    if set(changed.splitlines()) != CODE_FILES:
        raise RuntimeError('candidate code changes do not match the reviewed repair files')
    subprocess.run(['git', 'merge-base', '--is-ancestor', FIX, 'HEAD'], cwd=ROOT, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', FIX, 'HEAD'], cwd=ROOT, text=True)
    if set(changed.splitlines()) != CI_FILES:
        raise RuntimeError('validation branch contains code not included in the candidate commit')
    subprocess.run(['git', 'diff', '--check', BASE, 'HEAD'], cwd=ROOT, check=True)
    workflow = upstream_workflow()
    matrix = workflow['jobs']['test']['strategy']['matrix']
    versions = [str(row['torch']) for row in matrix['include']]
    if versions != VERSIONS:
        raise RuntimeError('upstream matrix changed; review the validation gate: ' + repr(versions))
    output('matrix', matrix)
    output('env', {key: str(value) for key, value in workflow['env'].items()})
    print('Full upstream matrix:', versions, flush=True)
    print('Candidate commit:', FIX, flush=True)


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
    failed = {case.get('name') for case in cases
              if case.find('failure') is not None or case.find('error') is not None}
    if failed:
        raise RuntimeError('failed test results: ' + repr(sorted(failed)))
    passed = {case.get('name') for case in cases
              if case.find('skipped') is None and case.get('status') != 'notrun'}
    if not set(required).issubset(passed):
        raise RuntimeError('mandatory frontend/native cases missing or skipped: ' + repr(sorted(set(required) - passed)))
    print('Checked report:', path.name, 'total=', len(cases), 'non-skipped=', len(passed), flush=True)


def run_command(command, env, cwd=ROOT):
    if sys.platform != 'win32':
        subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', command], cwd=cwd, env=env, check=True)
        return
    # Upstream's Windows run steps use native PowerShell, not WSL's bash.
    # A script file preserves embedded Python quotes and paths with spaces.
    with tempfile.TemporaryDirectory(prefix='pnnx-fork-ci-') as directory:
        script = Path(directory) / 'step.ps1'
        script.write_text(
            "$ErrorActionPreference = 'Stop'\n"
            "if ($PSVersionTable.PSVersion -lt [version]'7.3') { throw 'PowerShell 7.3 or newer is required' }\n"
            "$PSNativeCommandUseErrorActionPreference = $true\n"
            + command + "\n"
            "if (Test-Path variable:LASTEXITCODE) { exit $LASTEXITCODE }\n",
            encoding='utf-8')
        subprocess.run(['pwsh', '-NoLogo', '-NoProfile', '-NonInteractive', '-File', str(script)],
                       cwd=cwd, env=env, check=True)


def shell_self_test():
    def python_command(code):
        if sys.platform == 'win32':
            return "& '" + sys.executable.replace("'", "''") + "' -c '" + code.replace("'", "''") + "'"
        return shlex.join([sys.executable, '-c', code])

    with tempfile.TemporaryDirectory(prefix='pnnx shell regression ') as directory:
        directory = Path(directory)
        env = os.environ.copy()
        write_marker = python_command("from pathlib import Path; Path('marker').write_text('ok')")
        run_command(write_marker, env, cwd=directory)
        if (directory / 'marker').read_text() != 'ok':
            raise RuntimeError('native shell did not execute the Python command correctly')
        (directory / 'marker').unlink()
        try:
            run_command(python_command('import sys; sys.exit(7)') + '\n' + write_marker,
                        env, cwd=directory)
        except subprocess.CalledProcessError:
            pass
        else:
            raise RuntimeError('native shell swallowed an earlier command failure')
        if (directory / 'marker').exists():
            raise RuntimeError('native shell continued after an earlier command failure')
    print('Native shell success, quoting and early-failure regressions passed.', flush=True)


def run_steps(job, names):
    identity()
    workflow = upstream_workflow()
    matrix = json.loads(os.environ.get('MATRIX_JSON', '{}'))
    env = os.environ.copy()
    env.update({key: str(value) for key, value in workflow['env'].items()})
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
    env['EXTRA_CMAKE_ARGS'] = '-DNCNN_VULKAN=OFF'
    for key, value in workflow['jobs'][job].get('env', {}).items():
        env[key] = render(str(value), env, matrix)
    if 'PYTHONUSERBASE' in env:
        scheme = 'nt_user' if sys.platform == 'win32' else 'posix_user'
        user_scripts = sysconfig.get_path('scripts', scheme=scheme, vars={'userbase': env['PYTHONUSERBASE']})
        env['PATH'] = user_scripts + os.pathsep + env['PATH']
        # Plain Python steps (DLL setup and helper tests) need the same user site.
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write('PYTHONUSERBASE=' + env['PYTHONUSERBASE'] + '\n')
    reports = ROOT / 'fork-ci-reports'
    reports.mkdir(exist_ok=True)
    for name in names:
        matches = [step for step in workflow['jobs'][job]['steps'] if step.get('name') == name]
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
            if required != REQUIRED_PT2_TESTS:
                raise RuntimeError(
                    'required frontend/native test registration changed; missing=' +
                    repr(sorted(REQUIRED_PT2_TESTS - required)) + ' unexpected=' +
                    repr(sorted(required - REQUIRED_PT2_TESTS)))
            print('Required PT2 tests:', sorted(required), flush=True)
        print('::group::' + job + '/' + name, flush=True)
        print(command, flush=True)
        run_command(command, step_env)
        if has_ctest:
            check_report(report, required=required)
        print('::endgroup::', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['metadata']:
        metadata()
    elif sys.argv[1:] == ['identity']:
        identity()
    elif sys.argv[1:] == ['self-test']:
        shell_self_test()
    elif len(sys.argv) >= 4 and sys.argv[1] == 'run':
        run_steps(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit('usage: pnnx_fork_ci.py metadata|identity|self-test|run JOB STEP...')
