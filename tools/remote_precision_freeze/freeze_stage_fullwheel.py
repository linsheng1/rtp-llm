"""Offline binding/argv export for the existing C3 stage flow. No SSH or RPC."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent
CONTROL = '/home/admin/linshuo.zjl/publication-precision-20261101/control'
HOSTS = {'P': '10.44.12.175', 'D': '10.44.61.71', 'receiver': '10.44.61.73'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    path = Path(path)
    need(not path.is_symlink() and path.is_file(), 'regular non-symlink file required: ' + str(path))
    before = path.stat()
    need(before.st_size <= 4 * 1024**2, 'bounded control metadata only')
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'file changed while reading')
    return raw, {'path': str(path.resolve()), 'bytes': len(raw), 'SHA256': digest(raw)}


def doc(path):
    raw, pin = read(path)
    return json.loads(raw), pin


def option(argv, key):
    need(argv.count(key) == 1, 'missing or duplicate option: ' + key)
    i = argv.index(key)
    need(i + 1 < len(argv) and not argv[i + 1].startswith('--'), 'missing option value: ' + key)
    return argv[i + 1]


def literals(path):
    result = {}
    for node in ast.parse(read(path)[0]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def remote_path(value):
    path = PurePosixPath(value)
    need(path.is_absolute() and '..' not in path.parts and
         path.is_relative_to(PurePosixPath(CONTROL)), 'path outside existing control root')
    return value


def build(recipe):
    """Read safe existing receipts; never import deployment modules."""
    if recipe.get('flow') == 'state-source':
        from state_source_flow import build as build_source
        return build_source(recipe)
    if recipe.get('flow') == 'capture':
        from capture_flow import build_capture
        return build_capture(recipe)
    bench = Path(recipe['bench'])
    action, tag = recipe['action'], recipe['tag']
    need(action in ['baseline', 'd-restore', 'p-seed', 'p-restore'], 'unsupported stage')
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,40}', tag), 'invalid new batch tag')
    timeout = recipe['request_timeout_seconds']
    need(type(timeout) is int and 0 < timeout <= 180, 'request timeout outside existing budget')
    files, roles = {}, {}
    source, files['source_receipt'] = doc(recipe['source_receipt'])
    core = literals(bench / recipe['helpers']['core'])
    diag = literals(bench / recipe['helpers']['diagnostic'])
    need(core['B1'] == recipe['candidate_commit'], 'candidate differs from existing core')
    need(source['source_SHA256'] == diag['SOURCE_SHA'], 'source differs from existing diagnostic')
    need(source['R_tokens'] == 7169, 'unexpected frozen input size')
    for name, helper in recipe['helpers'].items():
        need(PurePosixPath(helper).name == helper, 'helper must be a local basename')
        raw, files[name] = read(bench / helper)
        if helper.endswith('.py'):
            ast.parse(raw)
    receiver_text = read(files['receiver']['path'])[0].decode()
    sender_text = read(files['sender']['path'])[0].decode()
    for name in ['core', 'diagnostic']:
        need(recipe['helpers'][name] in receiver_text and files[name]['SHA256'] in receiver_text,
             'receiver has different frozen dependency: ' + name)
    need(source['source_SHA256'] in receiver_text and source['source_SHA256'] == diag['SOURCE_SHA'],
         'receiver source binding differs')
    need(recipe['helpers']['core'] in sender_text and files['core']['SHA256'] in sender_text,
         'sender core binding differs')
    need(literals(files['sender']['path'])['PROOF_SHA'] == files['fresh_proof']['SHA256'] and
         '/all-probes' in sender_text and '/all-probes' in receiver_text,
         'fresh-proof helper/barrier differs from the existing corrected flow')
    for helper, expected in diag['PINS'].items():
        _, pin = read(bench / helper)
        need(pin['SHA256'] == expected, 'diagnostic dependency SHA differs: ' + helper)
        files['dependency:' + helper] = pin
    for role in ['P', 'D']:
        row, files['dispatch:' + role] = doc(recipe['roles'][role]['dispatch'])
        argv = row['argv']
        need(option(argv, '--role') == role and option(argv, '--host') == HOSTS[role], 'role/host differs')
        need(option(argv, '--prefill-host') == HOSTS['P'], 'prefill host differs')
        root = remote_path(option(argv, '--root'))
        need(PurePosixPath(root).name == row['root'], 'dispatch root differs')
        need(option(argv, '--env-sha256') == (core['ENV'][HOSTS[role]] if isinstance(core['ENV'],dict) else core['ENV']), 'environment SHA differs')
        need(option(argv, '--manifest-sha256') == core['MANIFESTS'][HOSTS[role]], 'manifest SHA differs')
        model_helper = next(v for v in argv if v.startswith(CONTROL + '/') and v.endswith('.py'))
        _, pin = read(bench / PurePosixPath(model_helper).name)
        need(pin['SHA256'] == core['MODEL_HELPER'], 'model helper SHA differs from core')
        files['model_helper:' + role] = pin
        need(re.fullmatch(r'[0-9a-f]{64}', row['plan_SHA256']), 'invalid model plan SHA')
        roles[role] = dict(host=HOSTS[role], root=root, plan_SHA256=row['plan_SHA256'],
                           namespace=option(argv, '--namespace'),
                           parent_controller_status=remote_path(recipe['roles'][role]['parent_controller_status']))
    same_namespace = roles['P']['namespace'] == roles['D']['namespace']
    need(same_namespace if action == 'd-restore' else not same_namespace if action in ['p-seed', 'p-restore'] else True,
         'action conflicts with actual P/D namespaces')
    interpreter = '/opt/conda310/bin/python3.10'
    recv = [interpreter, '-B', '-S', CONTROL + '/' + recipe['helpers']['receiver'],
            '--root', CONTROL + '/dgdet-stage-' + action + '-' + tag,
            '--action', action, '--tag', tag, '--seconds', '900',
            '--request-timeout-seconds', str(timeout)]
    if action == 'p-restore':
        seed, files['p_seed_receipt'] = doc(recipe['p_seed_receipt'])
        need(seed['stage']['summary']['input_ids_sha256'] == source['R_input_ids_sha256'], 'P seed uses different R')
        for key, flag in [('Pseed', '--p-seed-result'), ('manager', '--p-manager-report')]:
            ref = seed[key]
            recv += [flag, remote_path(ref['path']), flag + '-sha256', ref['SHA256']]
    jobs = {'receiver': dict(host=HOSTS['receiver'], argv=recv)}
    for role, row in roles.items():
        argv = [interpreter, '-B', '-S', CONTROL + '/' + recipe['helpers']['sender'], '--role', role,
                '--model-root', row['root'], '--plan-sha256', row['plan_SHA256'],
                '--output-root', CONTROL + '/dgdet-probe-' + role + '-' + action + '-' + tag,
                '--parent-controller-status', row['parent_controller_status']]
        if role == 'P' and action.endswith('restore'):
            argv += ['--fresh-proof']
        jobs[role] = dict(host=row['host'], argv=argv)
    return dict(schema='existing-c3-stage-freeze-v1', scope='OFFLINE_CONTROL_BINDING_ONLY',
                cluster='cluster3', candidate_commit=core['B1'], action=action, tag=tag, roles=roles,
                input=dict(tokens=source['R_tokens'], SHA256=source['R_input_ids_sha256'],
                           source_path=remote_path(source['source_path']), source_SHA256=source['source_SHA256']),
                runtime_gates='existing device identity/cache/lease/actual-CLI gates remain mandatory',
                fresh_proof_max_age_seconds=75, local_pins=files, jobs=jobs)


def validate(plan):
    need(plan.get('schema') == 'existing-c3-stage-freeze-v1', 'unsupported frozen plan')
    for pin in plan['local_pins'].values():
        _, actual = read(pin['path'])
        need(actual == pin, 'pinned file drift: ' + pin['path'])
    # Reuse the actual argparse declarations, without importing deployment code
    # or executing module-level servers, UID checks, probes, or RPCs.
    for role, job in plan['jobs'].items():
        key = 'receiver' if role == 'receiver' else 'sender'
        tree = ast.parse(read(plan['local_pins'][key]['path'])[0])
        declarations = []
        # Only literal argparse declarations, including those in main(). No import/main execution.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'p' and
                isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and
                isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'argparse' and
                    node.value.func.attr == 'ArgumentParser'):
                declarations.append(node)
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                  isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'add_argument' and
                  isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'p'):
                declarations.append(node)
        declarations.sort(key=lambda node: (node.lineno, node.col_offset))
        need(declarations, 'actual CLI declarations not found: ' + key)
        scope = {'argparse': argparse, 'Path': Path}
        exec(compile(ast.Module(body=declarations, type_ignores=[]), '<pinned-cli-declarations>', 'exec'), scope)
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                scope['p'].parse_args(job['argv'][4:])
        except SystemExit as error:
            raise ValueError('actual CLI argv rejected for ' + role + ': ' + stderr.getvalue().strip()) from error


def save_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + '.partial-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        need(temporary.read_bytes() == raw, 'plan verification failed before publication')
        os.link(temporary, path)  # atomic publication; an existing destination raises
    finally:
        temporary.unlink()
    return digest(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--recipe', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    for name in ['check', 'render']:
        item = sub.add_parser(name)
        item.add_argument('--plan', type=Path, required=True)
        item.add_argument('--sha256', required=True)
        if name == 'render':
            item.add_argument('--role', choices=['receiver', 'P', 'D'], required=True)
        else:
            item.add_argument('--cli-help', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        recipe, recipe_pin = doc(args.recipe)
        plan = build(recipe)
        plan['local_pins']['recipe'] = recipe_pin
        plan['local_pins']['freeze_stage'] = read(__file__)[1]
        if recipe.get('launcher'):
            need(recipe['launcher']=='launch_state_routes.py','unknown bounded route launcher')
            plan['local_pins']['route_launcher'] = read(BASE/recipe['launcher'])[1]
        if recipe.get('flow') == 'state-source':
            plan['local_pins']['state_source_flow.py'] = read(BASE / 'state_source_flow.py')[1]
        if recipe.get('flow') == 'capture':
            for name in ['capture_flow.py', 'launch_stage.py', 'new_capture_recipe.py', 'prepare_capture_models.py', 'bind_capture_controls.py']:
                plan['local_pins'][name] = read(BASE / name)[1]
        validate(plan)
        print(json.dumps(dict(state='FROZEN_OFFLINE_ONLY', SHA256=save_new(args.output, plan), model_requests=0)))
        return
    plan, pin = doc(args.plan)
    need(pin['SHA256'] == args.sha256, 'frozen plan SHA differs')
    validate(plan)
    if args.command == 'render':
        print(json.dumps(plan['jobs'][args.role], ensure_ascii=False))
    else:
        if args.cli_help:
            for key in ['receiver', 'sender']:
                result = subprocess.run([sys.executable, '-B', plan['local_pins'][key]['path'], '--help'],
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
                need(result.returncode == 0, 'actual CLI help failed: ' + key)
        print(json.dumps(dict(state='OFFLINE_PINS_AND_ARGV_PASS', CLI_help_checked=args.cli_help,
                              device_prepare_executed=False, model_requests=0)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(state='BLOCKED_OFFLINE_ZERO_RPC', reason=str(error), model_requests=0)))
        raise SystemExit(1)
