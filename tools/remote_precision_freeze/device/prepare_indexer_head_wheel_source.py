"""Source-only Git preparation; no build/cache/SDK/model access. Controller <=300s."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

TARGET = 'fb09988a5c552f4f5ff716472d8b50647af22d2c'
TASK = 'publication-precision-20261101'
FORK = 'https://github.com/linsheng1/rtp-llm'


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('root', 'old-source', 'new-source'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-old-head', required=True)
    a = p.parse_args()
    assert all(x.is_absolute() for x in (a.root, a.old_source, a.new_source))
    assert not a.root.exists() and not a.root.is_symlink()
    assert not a.new_source.exists() and not a.new_source.is_symlink()
    assert a.old_source.is_dir() and not a.old_source.is_symlink()
    task_base = a.root.parent.parent.resolve()
    assert task_base in a.new_source.parents and a.new_source.parent.resolve() == a.new_source.parent
    assert a.new_source.parent.stat().st_uid == os.getuid()
    assert a.expected_old_head in ('c1b4b19d82f56507f89be86dddf90497501d32a6',
                                   'ff13c95c5e6ce44cddad6789c4eb6acf2758a2e8',
                                   'f6af0bbd6133dcc47a48411c832b65d2f678da9e','b1d8f4e984ddfc7df83a796f44752d50923fdb50')
    assert a.root.parent.stat().st_uid == os.getuid()
    assert (a.root.parent / '.owner').read_text().strip() == TASK
    os.umask(0o077); a.root.mkdir(mode=0o700); (a.root / '.owner').write_text(TASK)
    deadline = time.monotonic() + 290
    result = dict(schema='cluster23-headfb09988-source-only-v1', state='RUNNING', commands=[], started_at=time.time(),
                  old_source=str(a.old_source), new_source=str(a.new_source), build_invoked=False,
                  tool_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def emit(row):
        print(json.dumps(row), flush=True)
    def run(stage, args, source=None):
        argv = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false']
        if source is not None:
            argv += ['-C', str(source)]
        argv += args
        left = min(120, deadline - time.monotonic()); assert left > 0, 'overall deadline reached'
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0'))
        stat_text = Path('/proc/' + str(child.pid) + '/stat').read_text() if sys.platform.startswith('linux') else None
        row = dict(stage=stage, argv=argv, PID=child.pid, UID=os.getuid(), started_at=time.time(), returncode=None,
                   start_ticks=stat_text[stat_text.rfind(')') + 2:].split()[19] if stat_text else None)
        result['commands'].append(row); emit(dict(event='STARTED', **row))
        try:
            out, err = child.communicate(timeout=left)
        except subprocess.TimeoutExpired:
            row['timed_out'] = True
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: out, err = child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); out, err = child.communicate(timeout=2)
        (a.root / (stage + '.stdout')).write_bytes(out); (a.root / (stage + '.stderr')).write_bytes(err)
        row.update(returncode=child.returncode, finished_at=time.time(), stdout_SHA256=hashlib.sha256(out).hexdigest(),
                   stderr_SHA256=hashlib.sha256(err).hexdigest()); emit(dict(event='EXITED', **row))
        assert child.returncode == 0 and not row.get('timed_out'), stage + ' failed; see own logs'
        return out
    code = 1
    try:
        before_head = run('old-head-before', ['rev-parse', 'HEAD'], a.old_source)
        before_status = run('old-status-before', ['status', '--porcelain=v1', '-z', '--untracked-files=all'], a.old_source)
        assert before_head.decode().strip() == a.expected_old_head, 'old HEAD differs'
        run('clone', ['clone', '--no-checkout', '--no-hardlinks', str(a.old_source), str(a.new_source)])
        run('set-origin', ['remote', 'set-url', 'origin', FORK], a.new_source)
        run('fetch', ['fetch', '--no-tags', 'origin', 'codex/publication-precision-20261101'], a.new_source)
        assert run('target-object', ['rev-parse', TARGET + '^{commit}'], a.new_source).decode().strip() == TARGET
        run('checkout', ['checkout', '--detach', TARGET], a.new_source)
        assert run('new-head', ['rev-parse', 'HEAD'], a.new_source).decode().strip() == TARGET
        assert not run('new-status', ['status', '--porcelain=v1', '-z', '--untracked-files=all'], a.new_source), 'new tree dirty'
        assert run('old-head-after', ['rev-parse', 'HEAD'], a.old_source) == before_head
        assert run('old-status-after', ['status', '--porcelain=v1', '-z', '--untracked-files=all'], a.old_source) == before_status
        result.update(state='SOURCE_PREPARED_NOT_BUILT', source_commit=TARGET, new_source_clean=True,
                      old_HEAD_and_status_bytes_unchanged=True, old_LFS_contents_verified=False)
        code = 0
    except Exception as error:
        result.update(state='FAILED', error=dict(type=type(error).__name__, message=str(error)))
    result.update(actual_returncode=code, finished_at=time.time())
    partial = a.root / 'source-result.json.partial'; final = a.root / 'source-result.json'
    with partial.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(str(partial), str(final)); partial.unlink()
    emit(dict(state=result['state'], actual_returncode=code, path=str(final),
              bytes=final.stat().st_size, SHA256=hashlib.sha256(final.read_bytes()).hexdigest()))
    return code


if __name__ == '__main__':
    sys.exit(main())
