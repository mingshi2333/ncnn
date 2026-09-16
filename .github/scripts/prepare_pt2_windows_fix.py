#!/usr/bin/env python3
"""Prepare reviewed code and CI commits on NEW fork branches only."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
BASE = '7af5e93d10df7cdc05a776a92e3559fed52b3cfa'
PARENT = 'b980798c22c3acb6aca30d7eea43aad54a857789'
OLD_VALIDATION = '160ba100bad3a29df2961d6adfba84362b7949db'
CODE_BRANCH = 'fix/pt2-python-paths-20260916'
CI_BRANCH = 'fix/pt2-python-paths-ci-20260916'
REPAIR_FILES = ['tools/pnnx/src/ir.cpp', 'tools/pnnx/tests/test_exported_program_parameter_roundtrip.cpp',
                'tools/pnnx/tests/test_python_codegen_paths.py', 'tools/pnnx/tests/CMakeLists.txt']
CODE_FILES = sorted(REPAIR_FILES + ['tools/pnnx/tests/pnnx_test_utils.py', 'tools/pnnx/tests/test_pnnx_test_utils.py',
                                  'tools/pnnx/tests/ncnn/test_attribute.py'])
CI_FILES = ['.github/scripts/pnnx_fork_ci.py', '.github/workflows/pnnx-fork-validation.yml']
ESCAPE = r'''static std::string escape_python_string(const std::string& value)
{
    std::string escaped;
    for (unsigned char ch : value)
    {
        if (ch == '\\' || ch == '\'' || ch == '"')
        {
            escaped += '\\';
            escaped += ch;
        }
        else if (ch < 0x20 || ch == 0x7f)
        {
            char hex[5];
            snprintf(hex, sizeof(hex), "\\x%02x", (unsigned int)ch);
            escaped += hex;
        }
        else
        {
            escaped += ch;
        }
    }
    return escaped;
}

'''
MODE = r'''    if (argc == 4 && strcmp(argv[1], "--python-paths") == 0)
    {
        pnnx::Graph graph;
        if (graph.parse("7767517\n2 1\npnnx.Input input 0 1 x #x=(1)f32\npnnx.Output output 1 0 x\n") != 0)
            return 1;
        return graph.python(argv[2], argv[3], {}, pnnx::get_model_stat(graph), true);
    }
'''


def command(*args, cwd=None, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('unexpected source context: ' + repr(old[:120]))
    return source.replace(old, new, 1)


def compile_helper(root, output):
    src = root / 'tools/pnnx/src'
    files = [root / 'tools/pnnx/tests/test_exported_program_parameter_roundtrip.cpp']
    files += [src / name for name in ('ir.cpp', 'model_stat.cpp', 'storezip.cpp', 'utils.cpp')]
    subprocess.run(['g++', '-std=c++11', '-O0', '-I' + str(src), *map(str, files), '-o', str(output)], check=True)


def main():
    workspace = Path(os.environ['GITHUB_WORKSPACE'])
    root = workspace / 'clean'
    env = os.environ.copy()
    auth = base64.b64encode(('x-access-token:' + env['GH_TOKEN']).encode()).decode()
    env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
               GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + auth)
    subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', 'https://github.com/mingshi2333/ncnn.git', str(root)], env=env, check=True)
    for sha in (PARENT, OLD_VALIDATION):
        subprocess.run(['git', 'fetch', 'origin', sha], cwd=root, env=env, check=True)
    subprocess.run(['git', 'checkout', '--detach', PARENT], cwd=root, check=True)
    if command('git', 'ls-remote', 'origin', 'refs/heads/pnnx-exported-program', cwd=root, env=env).split()[0] != BASE:
        raise RuntimeError('upstream source branch moved; refusing to publish')
    for name in (CODE_BRANCH, CI_BRANCH):
        if command('git', 'ls-remote', 'origin', 'refs/heads/' + name, cwd=root, env=env):
            raise RuntimeError('destination branch already exists; refusing to overwrite ' + name)
    helper_path = root / REPAIR_FILES[1]
    helper_path.write_text(replace_once(helper_path.read_text(), 'int main(int argc, char** argv)\n{\n',
                                      'int main(int argc, char** argv)\n{\n' + MODE))
    test_path = root / REPAIR_FILES[2]
    test_path.write_bytes((HERE / 'pt2_codegen_paths_test.py').read_bytes())
    cmake_path = root / REPAIR_FILES[3]
    cmake = cmake_path.read_text()
    cmake += '''\n# Path spelling must survive both Windows and POSIX Python code generation.\nadd_test(\n    NAME test_python_codegen_paths\n    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_codegen_paths.py $<TARGET_FILE:pnnx_test_exported_program_parameter_roundtrip>\n)\nset_tests_properties(test_python_codegen_paths PROPERTIES LABELS "pt2;pt2_frontend")\n'''
    cmake_path.write_text(cmake)
    binary = workspace / 'path-helper'
    compile_helper(root, binary)
    negative = subprocess.run([sys.executable, str(test_path), str(binary), '-v'], capture_output=True, text=True)
    (workspace / 'regression-before.log').write_text(negative.stdout + negative.stderr)
    if negative.returncode == 0 or 'SyntaxError' not in negative.stderr:
        raise RuntimeError('new regression did not reproduce the unescaped Python codegen failure')
    ir_path = root / REPAIR_FILES[0]
    ir = ir_path.read_text()
    ir = replace_once(ir, 'int Graph::python(', ESCAPE + 'int Graph::python(')
    counts = {}
    lines = ir.splitlines(keepends=True)
    for name, expected in [('pnnxbinpath', 1), ('pypath', 4), ('exported_program_path', 1)]:
        count = 0
        for i, line in enumerate(lines):
            if 'fprintf(pyfp,' in line and name + '.c_str()' in line:
                lines[i] = line.replace(name + '.c_str()', 'escape_python_string(' + name + ').c_str()')
                count += 1
        if count != expected:
            raise RuntimeError('unexpected path emission count for %s: %d != %d' % (name, count, expected))
        counts[name] = count
    ir_path.write_text(''.join(lines))
    compile_helper(root, binary)
    positive = subprocess.run([sys.executable, str(test_path), str(binary), '-v'], capture_output=True, text=True)
    (workspace / 'regression-after.log').write_text(positive.stdout + positive.stderr)
    print(positive.stdout + positive.stderr, flush=True)
    positive.check_returncode()
    subprocess.run([str(binary)], cwd=workspace, check=True)
    subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
    subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], cwd=root, check=True)
    subprocess.run(['git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], cwd=root, check=True)
    subprocess.run(['git', 'add', '--', *REPAIR_FILES], cwd=root, check=True)
    if set(command('git', 'diff', '--cached', '--name-only', cwd=root).splitlines()) != set(REPAIR_FILES):
        raise RuntimeError('unexpected code changes')
    subprocess.run(['git', 'commit', '-m', 'pnnx escape Python path literals and regress Windows code generation'], cwd=root, check=True)
    fix = command('git', 'rev-parse', 'HEAD', cwd=root)
    if set(command('git', 'diff', '--name-only', BASE, fix, cwd=root).splitlines()) != set(CODE_FILES):
        raise RuntimeError('candidate boundary does not match reviewed source files')
    for path in CI_FILES:
        data = command('git', 'show', OLD_VALIDATION + ':' + path, cwd=root) + '\n'
        if path.endswith('.py'):
            data = replace_once(data, "FIX = '" + PARENT + "'", "FIX = '" + fix + "'")
            data, n = re.subn(r'CODE_FILES = \{.*?\}\nVERSIONS', 'CODE_FILES = set(' + repr(CODE_FILES) + ')\nVERSIONS', data, flags=re.S)
            if n != 1:
                raise RuntimeError('unexpected candidate file gate')
            data = replace_once(data, 'if len(required) < 12:', 'if len(required) < 13:')
        else:
            data = data.replace('fix/pt2-cloud-ci-20260916', CI_BRANCH).replace(PARENT, fix)
            start = data.index('  quick-test:\n')
            end = data.index('  test:\n', start)
            quick = data[start:end]
            windows = quick.replace('  quick-test:\n', '  windows-test:\n    name: quick-test (windows-latest)\n', 1)
            windows = windows.replace('runs-on: ${{ matrix.os }}', 'runs-on: windows-latest')
            strategy = '''    strategy:\n      fail-fast: false\n      matrix:\n        os: [ubuntu-latest, macos-latest, windows-latest]\n'''
            windows = replace_once(windows, strategy, '')
            windows = windows.replace('${{ matrix.os }}', 'windows-latest')
            quick = replace_once(quick, '    needs: prepare\n', '    needs: [prepare, windows-test]\n')
            quick = replace_once(quick, 'os: [ubuntu-latest, macos-latest, windows-latest]', 'os: [ubuntu-latest, macos-latest]')
            data = data[:start] + windows + quick + data[end:]
            data = replace_once(data, '  build:\n    needs: prepare\n', '  build:\n    needs: [prepare, windows-test, quick-test]\n')
            data = replace_once(data, '  codeql:\n    needs: prepare\n', '  codeql:\n    needs: [prepare, windows-test, quick-test]\n')
            data = replace_once(data, 'needs: [prepare, build, quick-test, test, codeql]', 'needs: [prepare, build, windows-test, quick-test, test, codeql]')
            data = data.replace('Required frontend and eight native PT2 tests', 'Required frontend, path regression and eight native PT2 tests')
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(data)
    import yaml
    workflow = yaml.safe_load((root / CI_FILES[1]).read_text())
    assert len(workflow['jobs']['quick-test']['strategy']['matrix']['os']) == 2
    assert workflow['jobs']['windows-test']['runs-on'] == 'windows-latest'
    assert 'windows-test' in workflow['jobs']['build']['needs']
    assert 'windows-test' in workflow['jobs']['all-green']['needs']
    subprocess.run([sys.executable, '-m', 'py_compile', str(root / CI_FILES[0]), str(test_path)], check=True)
    subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
    subprocess.run(['git', 'add', '--', *CI_FILES], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-m', 'ci: run real Windows gate before full fork validation'], cwd=root, check=True)
    validation = command('git', 'rev-parse', 'HEAD', cwd=root)
    if set(command('git', 'diff', '--name-only', fix, validation, cwd=root).splitlines()) != set(CI_FILES):
        raise RuntimeError('fork-only boundary mismatch')
    if command('git', 'ls-remote', 'origin', 'refs/heads/pnnx-exported-program', cwd=root, env=env).split()[0] != BASE:
        raise RuntimeError('upstream source branch moved before publish')
    for name in (CODE_BRANCH, CI_BRANCH):
        if command('git', 'ls-remote', 'origin', 'refs/heads/' + name, cwd=root, env=env):
            raise RuntimeError('destination branch appeared before publish')
    # Publish only the clean code branch. The connector creates the validation
    # branch after reviewing its unreferenced commit, so Actions never updates
    # a workflow-bearing ref or the upstream source branch.
    subprocess.run(['git', 'push', 'origin', fix + ':refs/heads/' + CODE_BRANCH],
                   cwd=root, env=env, check=True)
    if command('git', 'ls-remote', 'origin', 'refs/heads/' + CODE_BRANCH, cwd=root, env=env).split()[0] != fix:
        raise RuntimeError('published clean branch verification failed')

    def post(endpoint, payload):
        request = urllib.request.Request('https://api.github.com/repos/mingshi2333/ncnn/' + endpoint,
            data=json.dumps(payload).encode(), method='POST',
            headers={'Authorization': 'Bearer ' + env['GH_TOKEN'], 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    entries = []
    for path in CI_FILES:
        blob = post('git/blobs', {'content': (root / path).read_text(), 'encoding': 'utf-8'})
        entries.append(dict(path=path, mode='100644', type='blob', sha=blob['sha']))
    tree = post('git/trees', {'base_tree': command('git', 'rev-parse', fix + '^{tree}', cwd=root), 'tree': entries})
    if tree['sha'] != command('git', 'rev-parse', validation + '^{tree}', cwd=root):
        raise RuntimeError('remote validation tree differs from the reviewed local tree')
    validation = post('git/commits', {'message': 'ci: run real Windows gate before full fork validation',
                                    'tree': tree['sha'], 'parents': [fix]})['sha']
    result = dict(base=BASE, parent=PARENT, candidate=fix, validation=validation,
                  code_branch=CODE_BRANCH, validation_branch=CI_BRANCH, code_files=CODE_FILES,
                  ci_files=CI_FILES, escaped_call_sites=counts)
    (workspace / 'repair-provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    (workspace / 'repair.patch').write_text(command('git', 'diff', PARENT, fix, cwd=root) + '\n')
    subprocess.run(['git', 'archive', '-o', str(workspace / 'reviewed-source.tar'), fix, *CODE_FILES], cwd=root, check=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
