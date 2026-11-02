"""Independent, opt-in N7169 P diagnostic prototype. Stdlib-only at import.

No deployment/admission claim. Real RID/ARM/epoch and GPU contracts still need
an isolated reviewed pilot. Never changes original calls, dispatch or returns.
"""
import contextvars
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time

MODEL = 'rtp_llm.models_py.model_desc.deepseek_v41_base_model'
ATTENTION = 'rtp_llm.models_py.modules.dsv41.fp8.attention_v41'
ASYNC = 'rtp_llm.models_py.modules.dsv41.fp8._v41_cp_async'
METADATA = 'rtp_llm.models_py.modules.dsv41.fp8._v41_producer_metadata'
GETTER = 'c3_cache_store_rid_getter_candidate_v1'
GPU_LIMIT, DISK_LIMIT, BATCH_LIMIT = 16 * 1024**2, 64 * 1024**2, 512 * 1024**2
CASE_DISK_LIMIT = DISK_LIMIT // 2
_scope = contextvars.ContextVar('p_state_observer_v2', default=None)


class _PendingArm(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reserve_case_disk(batch_directory):
    """Each case reserves32MiB; exactly two cases cap a rank at64MiB."""
    import fcntl
    batch = Path(batch_directory)
    batch.mkdir(parents=True, exist_ok=True)
    with open(batch / 'reservation.lock', 'a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = batch / 'reserved.json'
        previous = json.loads(counter.read_text()) if counter.exists() else 0
        if type(previous) is not int or previous < 0 or previous + CASE_DISK_LIMIT > BATCH_LIMIT:
            raise RuntimeError('BATCH_DISK_BUDGET')
        temporary = batch / ('reserved.%d.partial' % os.getpid())
        temporary.write_bytes(canonical(previous + CASE_DISK_LIMIT))
        os.replace(temporary, counter)
        return previous + CASE_DISK_LIMIT


def evidence(config, phase, fields=None):
    """Small device-local installation metadata; never reads model tensors."""
    stat_path = Path('/proc/self/stat')
    start = stat_path.read_text().rsplit(')', 1)[1].split()[19] if stat_path.exists() else None
    value = {'schema': 'c3-p-state-observer-evidence-v2', 'phase': phase,
             'PID': os.getpid(), 'UID': os.getuid(), 'PPID': os.getppid(),
             'start_ticks': start, 'nonce': config['nonce'],
             'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'torch_already_loaded': 'torch' in sys.modules,
             'getter_already_loaded': GETTER in sys.modules,
             'runtime_admission_verified_by_this_marker': False}
    value.update(fields or {})
    directory = Path(config['evidence_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    name = '%s-%d-%s-%s.json' % (config['nonce'], os.getpid(), start, phase.replace('.', '_'))
    fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(canonical(value))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        raise
    return value


def validate(config):
    if config.get('role') != 'P' or config.get('p_cuda_graph') is not False or config.get('moedbg') != 0:
        raise ValueError('P_EAGER_MOEDBG0_REQUIRED')
    if config.get('positions') != [7167, 7168] or config.get('layers') != [2, 8]:
        raise ValueError('N7169_TWO_SOURCE_LAYERS_TWO_POSITIONS_REQUIRED')
    if config.get('hidden_width') != 5120 or config.get('state_width') != 1024:
        raise ValueError('EXACT_GEOMETRY_REQUIRED')
    if config.get('expected_prefix_tokens') not in (0, 7168):
        raise ValueError('FULL_OR_SUFFIX1_PROFILE_REQUIRED')
    if not 0 < config.get('worker_timeout_seconds', 120) <= 120:
        raise ValueError('FINITE_WORKER_REQUIRED')
    for key in ('nonce', 'arm_path', 'identity_ref', 'input_ids_sha256', 'plan_sha256',
                'runtime_manifest_sha256', 'native_sha256', 'getter_path', 'getter_sha256',
                'output_dir', 'batch_budget_dir', 'evidence_dir', 'host'):
        if not config.get(key):
            raise ValueError('MISSING_' + key)
    nonce = config['nonce']
    if not 32 <= len(nonce) <= 64 or any(c not in '0123456789abcdef' for c in nonce):
        raise ValueError('INVALID_NONCE')
    for key in ('input_ids_sha256', 'plan_sha256', 'runtime_manifest_sha256', 'getter_sha256'):
        value = config[key]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('INVALID_SHA256_' + key)
    if not isinstance(config['native_sha256'], dict) or not config['native_sha256']:
        raise ValueError('NATIVE_PINS_REQUIRED')
    for value in config['native_sha256'].values():
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('INVALID_NATIVE_SHA256')
    return config


def read_small_canonical(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        file_stat = os.fstat(fd)
        if (file_stat.st_uid != os.getuid() or file_stat.st_size > 65536 or file_stat.st_nlink != 1
                or not stat.S_ISREG(file_stat.st_mode)):
            raise ValueError('ARM_FILE_IDENTITY_OR_SIZE')
        raw = os.read(fd, 65537)
    finally:
        os.close(fd)
    value = json.loads(raw)
    if canonical(value) != raw or len(raw) > 65536:
        raise ValueError('ARM_NOT_CANONICAL')
    return value, hashlib.sha256(raw).hexdigest()


def read_arm(config, now=None):
    """Both actual server identity and ARM may be pending at model startup."""
    now = time.time() if now is None else now
    found = read_small_canonical(config['identity_ref'])
    if found is None:
        return None
    identity, identity_sha = found
    if identity.get('schema') != 'c3-p-state-model-identity-v2' or identity.get('nonce') != config['nonce']:
        raise ValueError('INDEPENDENT_IDENTITY_SCHEMA_OR_NONCE')
    model = identity['model']
    if model['epoch']['host'] != config['host']:
        raise ValueError('INDEPENDENT_IDENTITY_HOST')
    if digest(model) != identity.get('model_identity_sha256'):
        raise ValueError('INDEPENDENT_IDENTITY_DIGEST')
    for key in ('plan_sha256', 'runtime_manifest_sha256', 'native_sha256'):
        if model[key] != config[key]:
            raise ValueError('INDEPENDENT_IDENTITY_STATIC_PIN_' + key)
    if len(identity.get('rank_epochs', [])) != 4:
        raise ValueError('FOUR_INDEPENDENT_RANK_EPOCHS_REQUIRED')
    if len({(rank['PID'], str(rank['start_ticks'])) for rank in identity['rank_epochs']}) != 4:
        raise ValueError('FOUR_DISTINCT_RANK_EPOCHS_REQUIRED')
    found = read_small_canonical(config['arm_path'])
    if found is None:
        return None
    arm, arm_sha = found
    if arm.get('schema') != 'c3-diagnostic-native-send-arm-v1':
        raise ValueError('ARM_SCHEMA')
    for key in ('nonce', 'input_ids_sha256'):
        if arm.get(key) != config[key]:
            raise ValueError('ARM_MISMATCH_' + key)
    if digest(arm['model']) != arm['model_identity_sha256']:
        raise ValueError('ARM_MODEL_DIGEST')
    if arm['model'] != model or arm['model_identity_sha256'] != identity['model_identity_sha256']:
        raise ValueError('ARM_NOT_BOUND_TO_INDEPENDENT_SERVER_IDENTITY')
    if arm['model']['epoch']['role'] != 'P' or type(arm.get('request_id')) is not int:
        raise ValueError('ARM_ROLE_OR_RID')
    if not now < arm.get('deadline_epoch', 0):
        raise ValueError('ARM_EXPIRED')
    # This source-bound profile supports exactly the full and suffix1 input.
    if arm.get('input_tokens') != 7169:
        raise ValueError('ARM_CASE_OUTSIDE_N7169_PROFILE')
    return arm, arm_sha, dict(identity, identity_ref_sha256=identity_sha)


class Snapshots:
    """Torch is obtained only after natural production/native imports."""
    def __init__(self, record):
        self.record = record
        self.torch = sys.modules['torch']

    def stream(self):
        current = self.torch.cuda.current_stream()
        return str(current.device), int(current.cuda_stream)

    def terminal(self):
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream())
        return event

    def _reserve(self, retained, transient):
        if self.record.retained + retained + transient > GPU_LIMIT:
            raise RuntimeError('GPU_TENSOR_BUDGET')
        self.record.retained += retained

    def _indices(self, positions, real_mask=None):
        torch = self.torch
        if positions is None or positions.ndim != 1 or not positions.numel():
            raise RuntimeError('ACTUAL_POSITIONS_UNAVAILABLE')
        wanted = torch.tensor([7167, 7168], device=positions.device, dtype=positions.dtype)
        matches = positions[:, None] == wanted[None, :]
        if real_mask is not None:
            if real_mask.shape != positions.shape or real_mask.dtype != torch.bool:
                raise RuntimeError('ACTUAL_REAL_MASK_ALIGNMENT')
            matches = matches & real_mask[:, None]
        return matches.to(torch.int64).argmax(0), matches.any(0)

    def copy(self, label, tensor, positions=None, real_mask=None):
        torch = self.torch
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (1, 2):
            raise RuntimeError('SNAPSHOT_TENSOR_UNAVAILABLE')
        if tensor.device.type != 'cuda' or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('CUDA_EAGER_ONLY')
        width = 1 if tensor.ndim == 1 else tensor.shape[1]
        required_width = (5120 if '.full_hidden5120' in label else
                          1024 if '.original_result1024' in label else
                          512 if '.original_values512' in label or '.original_scores512' in label else None)
        if required_width is not None and width != required_width:
            raise RuntimeError('POINT_WIDTH_NOT_EXPECTED_GEOMETRY')
        if width > 5120 or (width > 1024 and width != 5120):
            raise RuntimeError('SNAPSHOT_WIDTH_OUTSIDE_GEOMETRY')
        if positions is None or positions.shape[0] != tensor.shape[0]:
            raise RuntimeError('ORIGINAL_POSITION_ROW_ALIGNMENT')
        if width == 5120 and tensor.dtype != torch.bfloat16:
            raise RuntimeError('FULL_HIDDEN_BF16_REQUIRED')
        size = 2 * width * tensor.element_size()
        self._reserve(size + 64, positions.numel() * 32 + size * 8 + 4096)
        indices, found = self._indices(positions, real_mask)
        value = tensor.detach().index_select(0, indices).clone()
        value.masked_fill_(~found if value.ndim == 1 else ~found[:, None], 0)
        self.record.add(label, value, found, {'dtype': str(tensor.dtype),
            'original_shape': list(tensor.shape), 'sample_shape': list(value.shape),
            'full_width_preserved': True, 'real_mask_used': real_mask is not None})

    def pool(self, label, pool, slots, positions):
        torch = self.torch
        if (pool is None or pool.ndim != 2 or pool.shape[1] != 1024
                or slots is None or positions is None or slots.shape != positions.shape):
            raise RuntimeError('STATE_POOL_SLOT_POSITION_BINDING_MISSING')
        if pool.device.type != 'cuda' or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('CUDA_EAGER_ONLY')
        size = 2 * 1024 * pool.element_size()
        self._reserve(size + 64, slots.numel() * 32 + size * 8 + 4096)
        indices, found = self._indices(positions)
        chosen = slots.index_select(0, indices)
        valid = found & (chosen >= 0) & (chosen < pool.shape[0])
        value = pool.detach().index_select(0, chosen.clamp(0, pool.shape[0] - 1)).clone()
        value.masked_fill_(~valid[:, None], 0)
        self.record.add(label, value, valid, {'dtype': str(pool.dtype),
            'sample_shape': list(value.shape), 'full_width_preserved': True,
            'selection': 'actual_positions_to_original_slots',
            'invalid_owner_or_missing_mask_separate': True})


class Record:
    def __init__(self, owner, arm, arm_sha, identity_ref):
        self.owner, self.arm, self.arm_sha = owner, arm, arm_sha
        self.identity_ref = identity_ref
        self.samples, self.events, self.state_stack, self.launches = [], [], [], []
        self.tokens, self.previous_bindings, self.hits = [], [], {}
        self.problem = None
        self.retained = 0
        self.backend = owner.backend_factory(self)
        self.forward_stream = self.backend.stream()
        self.terminal_event = None
        self.forward_returned = False
        self.snapshots_released = False

    def guard(self, callback):
        if self.problem:
            return None
        try:
            return callback()
        except Exception as error:
            self.problem = type(error).__name__ + ':' + str(error)
            return None

    def add(self, label, value, mask, metadata):
        stream = self.backend.stream()
        launch_token = None
        if stream != self.forward_stream:
            if not self.launches:
                raise RuntimeError('SNAPSHOT_ON_UNTRACKED_STREAM')
            self.launches[-1]['snapshots'] = True
            self.launches[-1]['snapshot_labels'].append(label)
            launch_token = id(self.launches[-1])
        self.samples.append((label, value, mask, dict(metadata, stream=stream,
            producer_launch_token=launch_token,
            completion_chain='original_pending_event_wait' if launch_token else 'same_forward_stream')))

    def copy(self, *args, **kwargs):
        return self.guard(lambda: self.backend.copy(*args, **kwargs))

    def pool(self, *args):
        return self.guard(lambda: self.backend.pool(*args))

    def completion_ready_to_record(self):
        if self.problem:
            return False
        if not self.samples:
            self.problem = 'NOT_CAPTURED_NO_SNAPSHOT_SAMPLES'
            return False
        for token in self.tokens:
            if token['snapshots'] and (token.get('pending') is None or token.get('event') is None
                                      or token.get('joined_on') != self.forward_stream):
                self.problem = 'NOT_CAPTURED_SNAPSHOT_PENDING_NOT_JOINED'
                return False
        for layer in [2, 8]:
            required = ('_read_state' if self.owner.config['expected_prefix_tokens'] == 7168
                        else '_write_states')
            if not self.hits.get((layer, required)):
                self.problem = 'NOT_CAPTURED_EXPECTED_STATE_POINT_NOT_HIT'
                return False
        if self.backend.stream() != self.forward_stream:
            self.problem = 'NOT_CAPTURED_FORWARD_STREAM_CHANGED'
            return False
        return True


class Observer:
    def __init__(self, config, backend_factory=Snapshots, identity=None, scheduler=None, arm_reader=None):
        self.config = validate(json.loads(canonical(config)))
        self.backend_factory = backend_factory
        self.identity = identity or self.live_identity
        self.scheduler = scheduler or self.schedule
        self.arm_reader = arm_reader or read_arm
        self.consumed = False
        self.failed = False
        self.last_record = None
        self.status = 'WAITING_FOR_SEND_BOUNDARY_ARM'
        self.same_epoch_as = None

    def failure_evidence(self, record):
        try:
            evidence(self.config, 'capture-not-captured', {'problem': record.problem,
                'request_id': record.arm['request_id'], 'success_terminal_recorded': False})
        except Exception:
            pass

    def live_identity(self, model, inputs, arm, identity_ref):
        if getattr(model, 'kv_cache', None) is None:
            raise RuntimeError('WARMUP_NO_LIVE_CARRIER')
        self.check_epoch_ancestor(arm['model']['epoch'])
        own_start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        if not any(rank['PID'] == os.getpid() and rank['UID'] == os.getuid()
                   and str(rank['start_ticks']) == own_start for rank in identity_ref['rank_epochs']):
            raise RuntimeError('CURRENT_RANK_NOT_IN_INDEPENDENT_READY_IDENTITY')
        # Runtime/model artifact admission is an independent external interface;
        # matching these fields does not invent that audit or server isolation.
        path = Path(self.config['getter_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.config['getter_sha256']:
            raise RuntimeError('GETTER_SHA_MISMATCH')
        getter = sys.modules.get(GETTER)
        if getter is None:
            spec = importlib.util.spec_from_file_location(GETTER, path)
            getter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(getter)
            sys.modules[GETTER] = getter
        elif Path(getter.__file__).resolve() != path.resolve():
            raise RuntimeError('GETTER_ALREADY_LOADED_FROM_OTHER_PATH')
        torch = sys.modules['torch']
        tagged = inputs.attention_inputs
        if not isinstance(tagged, dict):
            tagged = {'single': tagged}
        if not tagged:
            raise RuntimeError('ATTENTION_TAGS_ABSENT')
        for tag, attention in tagged.items():
            if not attention.is_prefill:
                raise RuntimeError('NON_PREFILL_CALL')
            rid = getter.request_id(attention.cache_store_inputs)
            if (rid.device.type != 'cpu' or rid.dtype != torch.int64
                    or rid.ndim != 1 or rid.shape[0] != 1):
                raise RuntimeError('RID_NOT_CPU_INT64_SINGLE_REQUEST')
            if rid.tolist() != [arm['request_id']]:
                raise RuntimeError('ACTUAL_RID_NOT_ARMED_REQUEST')
        evidence(self.config, 'live-rid-bound', {'request_id': arm['request_id'],
            'independent_identity_ref_sha256': identity_ref['identity_ref_sha256'],
            'tag_count': len(tagged), 'getter_sha256': self.config['getter_sha256'],
            'actual_RID_CPU_int64_single': True})
        return True

    def check_epoch_ancestor(self, expected):
        """ARM names server root; observer executes inside spawned backend rank.

        Bind to exact root/ancestor epoch, never require rank PID==server PID.
        Runtime artifact audit and unique-RPC isolation remain external.
        """
        current = os.getpid()
        for _ in range(16):
            directory = Path('/proc') / str(current)
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if current == expected['PID']:
                if directory.stat().st_uid != expected['UID'] or fields[19] != str(expected['start_ticks']):
                    raise RuntimeError('ARM_SERVER_ROOT_EPOCH_MISMATCH')
                return
            current = int(fields[1])
            if current <= 1:
                break
        raise RuntimeError('RANK_NOT_DESCENDANT_OF_ARM_SERVER_EPOCH')

    def model_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(model, inputs, *args, **kwargs):
            if self.consumed or self.failed:
                return original(model, inputs, *args, **kwargs)
            try:
                pending = self.arm_reader(self.config)
                if pending is None:
                    raise _PendingArm()
                arm, arm_sha, identity_ref = pending
                if self.same_epoch_as is not None and arm['model'] != self.same_epoch_as():
                    raise RuntimeError('SECOND_CASE_NOT_SAME_SERVER_MODEL_EPOCH')
                if self.identity(model, inputs, arm, identity_ref) is not True:
                    raise RuntimeError('IDENTITY_UNRESOLVED')
                record = Record(self, arm, arm_sha, identity_ref)
            except _PendingArm:
                return original(model, inputs, *args, **kwargs)
            except Exception as error:
                self.status = 'NOT_CAPTURED:' + type(error).__name__ + ':' + str(error)
                self.failed = True
                try:
                    evidence(self.config, 'identity-not-captured', {'problem': self.status})
                except Exception:
                    pass
                return original(model, inputs, *args, **kwargs)
            self.consumed = True
            self.last_record = record
            token = _scope.set(record)
            try:
                result = original(model, inputs, *args, **kwargs)
            except BaseException:
                self.status = 'ORIGINAL_FORWARD_FAILED_NO_SUCCESS_EVENT'
                record.problem = self.status
                self.failure_evidence(record)
                raise
            finally:
                _scope.reset(token)
            record.forward_returned = True
            if not record.guard(record.completion_ready_to_record):
                self.status = 'NOT_CAPTURED:' + str(record.problem)
                self.failure_evidence(record)
                return result
            record.guard(lambda: setattr(record, 'terminal_event', record.backend.terminal()))
            if record.terminal_event is None:
                self.status = 'NOT_CAPTURED:' + str(record.problem)
                self.failure_evidence(record)
                return result
            self.status = 'TERMINAL_RECORDED_DRAIN_PENDING'
            try:
                self.scheduler(record)
            except Exception as error:
                record.problem = 'DRAIN_SCHEDULE_FAILED:' + str(error)
                self.status = 'NOT_CAPTURED:' + record.problem
            return result
        return wrapped

    def _eligible(self, record, attention):
        cp = getattr(attention, '_cp_ctx', None)
        lengths = getattr(cp, 'input_lengths_global_host', None)
        prefixes = getattr(cp, 'prefix_lengths_host', None)
        prefix = self.config['expected_prefix_tokens']
        case = (prefix, 7169 - prefix)
        evidence = {'point': 'prefill_eligibility', 'layer': attention.layer_id,
            'source_layer': getattr(attention, 'kv_source_layer_id', None),
            'ratio': getattr(attention, 'compress_ratio', None),
            'is_kv_source': getattr(attention, 'is_kv_source', None),
            'cp_size': getattr(cp, 'cp_size', None), 'cp_rank': getattr(cp, 'cp_rank', None),
            'sharded': getattr(cp, 'kv_cache_sharded', None),
            'lengths_host': lengths, 'prefixes_host': prefixes,
            'padded_seq_len': getattr(cp, 'padded_seq_len', None)}
        record.events.append(evidence)
        if (attention.compress_ratio != 2 or not attention.is_kv_source
                or cp is None or cp.cp_size != 4 or not cp.kv_cache_sharded
                or not isinstance(lengths, (list, tuple)) or tuple(lengths) != (case[1],)
                or not isinstance(prefixes, (list, tuple)) or tuple(prefixes) != (case[0],)):
            raise RuntimeError('BRANCH_OUTSIDE_SINGLE_N7169_PROFILE')
        return cp

    def attention_wrapper(self, name, original):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(attention, *args, **kwargs):
            record = _scope.get()
            if record is None or attention.layer_id not in self.config['layers'] or record.problem:
                return original(attention, *args, **kwargs)
            bound = record.guard(lambda: signature.bind(attention, *args, **kwargs).arguments)
            if bound is None:
                return original(attention, *args, **kwargs)
            label = str(attention.layer_id) + '.' + name
            record.hits[(attention.layer_id, name)] = record.hits.get((attention.layer_id, name), 0) + 1
            if name == '_prefill_produce':
                cp = record.guard(lambda: self._eligible(record, attention))
                if cp is not None:
                    record.copy(label + '.full_hidden5120', bound['x'],
                                positions=cp.global_positions, real_mask=cp.local_is_real)
                    for field in ('global_positions', 'local_is_real', 'req_id_per_token'):
                        value = getattr(cp, field, None)
                        if value is None:
                            record.events.append({'point': label, 'missing_optional_cp_field': field})
                        else:
                            record.copy(label + '.cp.' + field, value,
                                        positions=cp.global_positions, real_mask=cp.local_is_real)
                return original(attention, *args, **kwargs)
            if name == '_produce_global':
                record.events.append({'point': label, 'batched_groups': bound.get('batched_groups', False),
                    'previous_state_present': bound.get('previous_state') is not None,
                    'prefill': bound.get('prefill'), 'projected_full': bound.get('projected_full', False)})
                if bound.get('batched_groups', False) or bound.get('prefill') is not True:
                    record.problem = 'BRANCH_OUTSIDE_SINGLE_N7169_PROFILE'
                return original(attention, *args, **kwargs)
            region = sys.modules[ATTENTION].CSA_STATE
            if name == '_slots':
                result = original(attention, *args, **kwargs)
                if record.state_stack and bound['region'] == region:
                    record.state_stack[-1]['slots'] = result
                return result
            if name == '_source_pool':
                result = original(attention, *args, **kwargs)
                if record.state_stack and bound['region'] == region:
                    state = record.state_stack[-1]
                    state['pool'] = result
                    if state['slots'] is not None and not state['pre_copied']:
                        state['pre_copied'] = True
                        record.pool(state['label'] + '.raw_before', result, state['slots'], state['positions'])
                return result
            if name in ('_read_state', '_write_states'):
                positions, rows = bound['positions'], bound['req_ids']
                slots = bound.get('slots')
                if positions is None:
                    # Future multi-request profile may use this exact carrier;
                    # current single-case rejects such a branch before terminal.
                    match = next((b for b in record.previous_bindings if b['slots'] is slots), None)
                    if match is None:
                        record.problem = 'EXPLICIT_SLOTS_WITHOUT_ORIGINAL_STARTS_CARRIER'
                        return original(attention, *args, **kwargs)
                    positions, rows = match['positions'], match['rows']
                    record.events.append({'point': label, 'binding': 'natural_prepare_raw_original_starts',
                                          'original_positions_argument_is_None': True})
                state = {'slots': slots, 'positions': positions, 'pool': None,
                         'label': label, 'pre_copied': False}
                record.state_stack.append(state)
                record.copy(label + '.positions', positions, positions=positions)
                record.copy(label + '.batch_rows_NOT_RID', rows, positions=positions)
                if name == '_write_states':
                    record.copy(label + '.original_values512', bound['values'], positions=positions)
                    record.copy(label + '.original_scores512', bound['scores'], positions=positions)
                try:
                    result = original(attention, *args, **kwargs)
                    record.copy(label + '.original_slots', state['slots'], positions=positions)
                    if name == '_read_state':
                        record.copy(label + '.original_result1024_post_gather', result, positions=positions)
                    else:
                        record.pool(label + '.raw_after', state['pool'], state['slots'], positions)
                    return result
                finally:
                    record.state_stack.pop()
            return original(attention, *args, **kwargs)
        return wrapped

    def async_launch_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            record = _scope.get()
            if record is None:
                return original(*args, **kwargs)
            token = {'snapshots': False, 'pending': None, 'event': None,
                     'joined_on': None, 'snapshot_labels': []}
            record.tokens.append(token)
            record.launches.append(token)
            try:
                result = original(*args, **kwargs)
                token['pending'] = result
                token['event'] = getattr(result, 'event', None)
                record.guard(lambda: record.events.append({'point': 'original_async_launch_returned',
                    'pending_object': id(result), 'has_original_event': result.event is not None,
                    'original_event_object': id(result.event), 'launch_token': id(token),
                    'has_snapshots': token['snapshots'], 'snapshot_labels': list(token['snapshot_labels'])}))
                return result
            finally:
                record.launches.pop()
        return wrapped

    def pending_wait_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(pending, *args, **kwargs):
            result = original(pending, *args, **kwargs)
            record = _scope.get()
            if record is not None:
                def mark_join():
                    for token in record.tokens:
                        if token['pending'] is pending and pending.event is not None:
                            if pending.event is not token['event']:
                                raise RuntimeError('ORIGINAL_PENDING_EVENT_IDENTITY_CHANGED')
                            token['joined_on'] = record.backend.stream()
                            record.events.append({'point': 'original_pending_wait_returned',
                                'pending_object': id(pending), 'original_event_object': id(pending.event),
                                'launch_token': id(token), 'joined_on': token['joined_on']})
                record.guard(mark_join)
            return result
        return wrapped

    def metadata_wrapper(self, original):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            record = _scope.get()
            if record is not None:
                def bind():
                    if result is None or getattr(result, 'previous_slots', None) is None:
                        return
                    starts = signature.bind(*args, **kwargs).arguments['starts']
                    torch = record.backend.torch
                    record.previous_bindings.append({'slots': result.previous_slots,
                        'positions': (starts - 1).clamp_min(0),
                        'rows': torch.arange(starts.shape[0], device=starts.device)})
                    # This profile is single request, so actual prior-slots
                    # metadata means eligibility was not established correctly.
                    record.problem = 'FUSED_PREVIOUS_SLOTS_OUTSIDE_SINGLE_REQUEST_PROFILE'
                record.guard(bind)
            return result
        return wrapped

    def patch_module(self, name, module):
        targets = []
        if name == MODEL:
            targets = [(module.DeepSeekV4Model, 'forward', self.model_wrapper)]
        elif name == ATTENTION:
            targets = [(module.AttentionV41FP8, method,
                        lambda fn, m=method: self.attention_wrapper(m, fn))
                       for method in ('_prefill_produce', '_produce_global', '_read_state',
                                      '_write_states', '_slots', '_source_pool')]
        elif name == ASYNC:
            targets = [(module, 'launch', self.async_launch_wrapper),
                       (module.Pending, 'wait', self.pending_wait_wrapper)]
        elif name == METADATA:
            targets = [(module, 'prepare_raw', self.metadata_wrapper)]
        for owner, method, make in targets:
            original = getattr(owner, method)
            if not getattr(original, '_p_state_observer_v2', False):
                wrapper = make(original)
                wrapper._p_state_observer_v2 = True
                setattr(owner, method, wrapper)
        evidence(self.config, 'hooked.' + name, {'module': name,
            'module_file': getattr(module, '__file__', None),
            'module_sha256': (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                              if getattr(module, '__file__', None) else None),
            'methods': [method for _, method, _ in targets]})

    def schedule(self, record):
        worker = threading.Thread(target=self.worker, args=(record,),
                                  name='p-state-drain-v2', daemon=True)
        record.worker = worker
        worker.start()

    def worker(self, record, monotonic=time.monotonic, sleep=time.sleep):
        deadline = monotonic() + self.config.get('worker_timeout_seconds', 120)
        record.worker_deadline = deadline
        try:
            while not record.terminal_event.query():
                if monotonic() >= deadline:
                    raise RuntimeError('TERMINAL_EVENT_TIMEOUT')
                sleep(0.02)
            self.persist(record)
            record.samples.clear()
            record.tokens.clear()
            record.previous_bindings.clear()
            record.terminal_event = None
            record.retained = 0
            record.snapshots_released = True
            evidence(self.config, 'drain-finished', {'result': record.result,
                'snapshot_tensor_references_released': True,
                'actual_allocator_release_measured': False})
            self.status = 'CAPTURED_STATE_DIAGNOSTIC_PENDING_REVIEW'
        except Exception as error:
            record.problem = type(error).__name__ + ':' + str(error)
            self.status = 'NOT_CAPTURED:' + record.problem
            try:
                evidence(self.config, 'drain-failed', {'problem': record.problem})
            except Exception:
                pass

    def persist(self, record):
        """Called only by the worker after actual terminal Event.query=True."""
        if not record.forward_returned or record.terminal_event is None or not record.terminal_event.query():
            raise RuntimeError('ACTUAL_TERMINAL_COMPLETION_REQUIRED')
        root = Path(self.config['output_dir'])
        partial = root / (self.config['nonce'] + '-%d.partial' % os.getpid())
        final = root / (self.config['nonce'] + '-%d' % os.getpid())
        partial.mkdir(parents=True, exist_ok=False)
        reserve_case_disk(self.config['batch_budget_dir'])
        total, samples = 0, []
        torch = record.backend.torch
        for index, (label, tensor, mask, metadata) in enumerate(record.samples):
            if time.monotonic() >= record.worker_deadline:
                raise RuntimeError('BOUNDED_DRAIN_DEADLINE')
            cpu = tensor.detach().contiguous().cpu()
            raw = cpu.view(torch.uint8).numpy().tobytes()
            total += len(raw)
            if total + 1024**2 > CASE_DISK_LIMIT:
                raise RuntimeError('RANK_DISK_BUDGET')
            file = '%04d.raw' % index
            destination = partial / file
            destination.write_bytes(raw)
            if destination.stat().st_size != len(raw) or hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
                raise RuntimeError('SNAPSHOT_DISK_SIZE_OR_SHA')
            samples.append(dict(metadata, label=label, file=file, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), valid=mask.detach().cpu().tolist(),
                finite=torch.isfinite(cpu).tolist() if cpu.is_floating_point() else None))
        manifest = {'profile': 'P_STATE_DIAGNOSTIC_V2_NOT_ACCEPTANCE',
            'request_id': record.arm['request_id'], 'nonce': self.config['nonce'],
            'arm_sha256': record.arm_sha, 'input_ids_sha256': record.arm['input_ids_sha256'],
            'independent_identity_ref_sha256': record.identity_ref['identity_ref_sha256'],
            'model_identity_sha256_from_arm': record.arm['model_identity_sha256'],
            'server_epoch_artifact_admission_verified_by_this_helper': False,
            'server_single_rpc_isolation_verified_by_this_helper': False,
            'source_process_provenance_verified': False,
            'PID': os.getpid(), 'UID': os.getuid(),
            'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            'forward_stream': record.forward_stream, 'terminal_event_query_completed': True,
            'terminal_covers_cpp_lm_head_or_rpc_completion': False,
            'gpu_retained_tensor_bytes': record.retained, 'allocator_overhead_accounted': False,
            'disk_raw_bytes': total, 'positions': [7167, 7168], 'layers': [2, 8],
            'events': record.events, 'samples': samples, 'problem': record.problem}
        encoded = canonical(manifest)
        if len(encoded) > 1024**2:
            raise RuntimeError('METADATA_BUDGET')
        (partial / 'manifest.json').write_bytes(encoded)
        if final.exists():
            raise RuntimeError('OUTPUT_EXISTS')
        os.rename(partial, final)
        record.result = {'path': str(final), 'manifest_sha256': hashlib.sha256(encoded).hexdigest()}


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, observer):
        self.observer = observer

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (MODEL, ATTENTION, ASYNC, METADATA):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            raise ImportError('NATURAL_EXEC_MODULE_LOADER_REQUIRED')
        original, observer = spec.loader, self.observer
        class Loader:
            def create_module(self, spec):
                method = getattr(original, 'create_module', None)
                return method(spec) if method else None
            def exec_module(self, module):
                original.exec_module(module)
                observer.patch_module(fullname, module)
        spec.loader = Loader()
        return spec


class Dispatcher:
    """Exactly two frozen cases, short then full; no RPCs or automatic retries."""
    def __init__(self, config, observer_factory=Observer):
        if config.get('schema') != 'c3-p-state-observer-batch-v2' or len(config.get('cases', [])) != 2:
            raise ValueError('EXACTLY_TWO_FROZEN_CASES_REQUIRED')
        cases = config['cases']
        if [case.get('expected_prefix_tokens') for case in cases] != [7168, 0]:
            raise ValueError('SHORT_THEN_FULL_REQUIRED')
        if cases[0]['nonce'] == cases[1]['nonce'] or cases[0]['arm_path'] == cases[1]['arm_path']:
            raise ValueError('DISTINCT_NONCE_AND_ARM_REQUIRED')
        if cases[0]['identity_ref'] == cases[1]['identity_ref']:
            raise ValueError('DISTINCT_IDENTITY_REF_NONCES_REQUIRED')
        for key in ('host', 'plan_sha256', 'runtime_manifest_sha256', 'native_sha256',
                    'getter_path', 'getter_sha256', 'batch_budget_dir', 'input_ids_sha256'):
            if cases[0][key] != cases[1][key]:
                raise ValueError('SAME_EPOCH_STATIC_PIN_REQUIRED_' + key)
        self.observers = [observer_factory(case) for case in cases]
        self.observers[1].same_epoch_as = lambda: self.observers[0].last_record.arm['model']
        self.active = 0

    def model_wrapper(self, original):
        candidates = [case.model_wrapper(original) for case in self.observers]
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if _scope.get() is not None:
                return original(*args, **kwargs)
            first = self.observers[0]
            record = first.last_record
            if (self.active == 0 and first.status == 'CAPTURED_STATE_DIAGNOSTIC_PENDING_REVIEW'
                    and record is not None and record.snapshots_released and getattr(record, 'result', None)):
                self.active = 1
            case = self.observers[self.active]
            if case.consumed:
                return original(*args, **kwargs)
            return candidates[self.active](*args, **kwargs)
        return wrapped

    def patch_module(self, name, module):
        if name == MODEL:
            targets = [(module.DeepSeekV4Model, 'forward', lambda case, fn: case.model_wrapper(fn))]
        elif name == ATTENTION:
            targets = [(module.AttentionV41FP8, method,
                        lambda case, fn, m=method: case.attention_wrapper(m, fn))
                       for method in ('_prefill_produce', '_produce_global', '_read_state',
                                      '_write_states', '_slots', '_source_pool')]
        elif name == ASYNC:
            targets = [(module, 'launch', lambda case, fn: case.async_launch_wrapper(fn)),
                       (module.Pending, 'wait', lambda case, fn: case.pending_wait_wrapper(fn))]
        elif name == METADATA:
            targets = [(module, 'prepare_raw', lambda case, fn: case.metadata_wrapper(fn))]
        else:
            return
        for owner, method, factory in targets:
            original = getattr(owner, method)
            if getattr(original, '_p_state_observer_v2', False):
                raise RuntimeError('OBSERVER_ALREADY_PATCHED')
            if name == MODEL:
                wrapper = self.model_wrapper(original)
            else:
                callbacks = {id(case): factory(case, original) for case in self.observers}
                def route(*args, _original=original, _callbacks=callbacks, **kwargs):
                    record = _scope.get()
                    callback = _callbacks.get(id(record.owner)) if record else None
                    return (callback or _original)(*args, **kwargs)
                wrapper = functools.wraps(original)(route)
            wrapper._p_state_observer_v2 = True
            setattr(owner, method, wrapper)
        for case in self.observers:
            evidence(case.config, 'hooked.' + name, {'module': name,
                'methods': [method for _, method, _ in targets],
                'module_file': getattr(module, '__file__', None),
                'module_sha256': (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                  if getattr(module, '__file__', None) else None)})


def install(config):
    if any(name in sys.modules for name in (MODEL, ATTENTION, ASYNC, METADATA)):
        raise RuntimeError('BOOTSTRAP_TOO_LATE')
    observer = Dispatcher(config)
    sys.meta_path.insert(0, Finder(observer))
    for case in observer.observers:
        evidence(case.config, 'lazy-finder-installed', {'natural_targets': [MODEL, ATTENTION, ASYNC, METADATA],
                                             'final_RID_required_at_startup': False,
                                             'actual_model_epoch_required_at_startup': False})
    return observer
