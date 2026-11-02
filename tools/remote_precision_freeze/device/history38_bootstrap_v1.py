"""New opt-in startup profile, compatible with python -S.

Usage: HISTORY_CAPTURE_CONFIG=/device/config.json python -S this_file.py [server args]
No product or GPU import occurs before the production entry point naturally
executes. Spawn reruns this bootstrap; sitecustomize is deliberately unused.
"""
import importlib.util
import hashlib
import json
import multiprocessing.spawn
import os
from pathlib import Path
import sys

_BOOTSTRAP_PATH = str(Path(__file__).resolve())
_TARGET = os.environ.get('P_ONLY_OBSERVER_TARGET', 'rtp_llm.start_server')
if _TARGET != 'rtp_llm.start_server' and os.environ.get('P_ONLY_OBSERVER_CPU_TEST') != '1':
    raise RuntimeError('PRODUCTION_TARGET_IS_FIXED')


def _install_spawn_reentry():
    original = multiprocessing.spawn.get_preparation_data
    if getattr(original, '_p_observer_v4', False):
        return

    def preparation(name):
        data = original(name)
        data.pop('init_main_from_name', None)
        data['init_main_from_path'] = _BOOTSTRAP_PATH
        return data
    preparation._p_observer_v4 = True
    multiprocessing.spawn.get_preparation_data = preparation


def _run_production():
    config_path = Path(os.environ['HISTORY_CAPTURE_CONFIG'])
    if config_path.is_symlink() or config_path.stat().st_size > 128*1024 or config_path.stat().st_uid != 500:
        raise RuntimeError('OWNED_BOUNDED_HISTORY_CONFIG_REQUIRED')
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['HISTORY_CAPTURE_CONFIG_SHA256']:
        raise RuntimeError('FROZEN_HISTORY_CONFIG_DRIFT')
    config = json.loads(raw)
    for name in ['history38_bootstrap_v1.py', 'history38_observer_v1.py', 'history38_latches_v1.py']:
        path = Path(_BOOTSTRAP_PATH).with_name(name)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != config['control_source_SHA256'][name]:
            raise RuntimeError('HISTORY_STARTUP_SOURCE_DRIFT')
    helper_path = str(Path(_BOOTSTRAP_PATH).with_name('history38_observer_v1.py'))
    helper_name = '_c3_history38_observer_startup_v1'
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper
    helper_spec.loader.exec_module(helper)
    observer = helper.install(config)
    # Exposed for a later reviewed completion-boundary integration; no automatic
    # draining/synchronizing, no arm-RPC creation, and no new service endpoint.
    globals()['_P_STATE_OBSERVER_V4'] = observer
    _install_spawn_reentry()
    # Match python -m's cwd entry rather than exposing helper directory as the
    # production import root. Spawn inherits this same production sys.path.
    if sys.path:
        sys.path[0] = os.getcwd()
    spec = importlib.util.find_spec(_TARGET)
    if spec is None or spec.loader is None or not hasattr(spec.loader, 'get_code'):
        raise RuntimeError('PRODUCTION_CODE_LOADER_REQUIRED')
    code = spec.loader.get_code(_TARGET)
    globals()['__file__'] = spec.origin
    globals()['__package__'] = spec.parent
    globals()['__spec__'] = spec
    sys.argv[0] = spec.origin
    # Current __main__/__mp_main__ namespace preserves target pickling. The
    # production main guard/NUMA setup runs only in the original main process.
    exec(code, globals(), globals())


if __name__ in ('__main__', '__mp_main__'):
    _run_production()
