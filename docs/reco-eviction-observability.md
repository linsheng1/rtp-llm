# HOST-to-DISK eviction observations

This observation-only increment is based on `c253560f4`. It does not change disk
scheduling, timeout, cache identity, admission, references, FROZEN recovery, or
remote publication. No production configuration is changed by this patch.

## Byte accounting

The existing `rtp_llm_kv_cache_transfer_bytes` remains successful logical-transfer
payload only. The eviction runner additionally reports:

| QPS metric | Meaning |
| --- | --- |
| `rtp_llm_kv_cache_transfer_attempted_bytes` | Requested payload of valid eviction dispatch attempts, including subsequent rejection or failure |
| `rtp_llm_kv_cache_transfer_failed_attempt_bytes` | The same requested payload when the logical transfer fails; zero on success |

Both retain operation, source/target tier, pool name and FULL/SWA group-type tags.
Existing transfer QPS and eviction total/success/fail/cancel counters provide the
attempt counts; no duplicate attempt counter is introduced. An attempt is recorded
at logical completion, not submission. Invalid runner descriptors and unresolved
group/pool sizes are excluded from byte totals, not treated as measured zero;
existing failure counts remain available. Load callers retain their existing
byte reporting.

These are payload bytes per logical dispatch, not summed rank bytes, physical
`pwrite` bytes, completed disk bytes, distinct data, or duplicate-write bytes.
A failed logical attempt may have performed no I/O, partial I/O, or I/O that
finishes after the RPC timeout. Integrate QPS over the same role/pool/time window;
do not multiply a sampled GAUGE average by the window duration.

`disk_cache_write_bytes` separately counts locally successful write batches,
including writes that finish after coordinator timeout. Take per-process deltas
with restart handling. It does not account all partial failed batches or prove
physical media persistence.

## Bounded per-rank disk probe

The probe is off by default. Set `RTP_LLM_DISK_IO_TRACE_BATCHES=N` before process
startup to select the first N HOST-to-DISK worker batches in that process, capped
at 128. Optional `RTP_LLM_DISK_IO_TRACE_ID` restricts selection to one execution
trace (zero or unset selects all eligible traces). Values are unsigned decimal;
invalid values disable the probe. INFO logging must be enabled. Settings are read once; changing a shell
environment does not enable tracing in an existing process.

Selection reserves a whole batch: up to three INFO lines (`disk_submit`,
`disk_complete` or `disk_rejected`, `worker_reply_ready`), at most 384 lines per
process. Unrelated directions and nonmatching traces do not consume the budget.
Normal success and failure use the same budget; no per-pwrite logging is added.

Each line includes execution trace, rank, group/pool, first path index and disk
block, descriptor count, requested stride bytes, result and steady-clock times.
For writes, existing executor timestamps provide enqueue, earliest worker start,
sort end and last accepted worker completion, plus syscall count/sum/max.
Trace + rank + group + first path/disk block identifies the worker sub-batch;
do not join by trace alone. Steady timestamps are comparable within a process,
not across hosts. Log timestamps provide wall-clock anchors. Log
`requested_stride_bytes` includes disk stride/padding; the attempted-byte metric
uses source-pool payload size. These quantities need not be equal.

`disk_submit` is an enqueue attempt, not acceptance proof. `disk_rejected` means
no disk worker accepted I/O. `disk_complete` runs after every accepted worker has
joined, even if the coordinator already timed out. `worker_reply_ready` follows
the before-complete callback and precedes context completion; it is not CQ
receipt or successful coordinator adoption. Zero worker timestamps mean no
worker timing evidence, not a measured zero-duration stage. Parallel I/O wall
time also includes later participants' queueing and joining.

A pre-acceptance submission exception can go directly from `disk_submit` to
failed `worker_reply_ready` through the existing exception handler. Missing
completion logs alone never prove that I/O stopped or permit resource release.

## Attribution limits

The cache currently has no persistent acquisition-origin field to reuse. This
patch deliberately does not label all DEVICE-origin data as locally computed or
all remote data as D-generated. Reliable attribution needs successful remote-load
adoption linked to logical key/group and resource generation, surviving tier
migration and being reset on replacement. That separate provenance change is not
included in this minimal observation increment. Remote origin and repeated
eviction are independent facts.

A small difference between disk-write accounting and remote-read bytes does not
prove the absence of retries or D influence. Compare HOST inflow/inventory,
attempt/success/failure by FULL/SWA, locally completed writes, and valid remote
read evidence in the same window. Missing P remote-byte samples are not zero.

## Validation status

Source inspection and `git diff --check` only. This increment has not been
compiled, run, or deployed. No tests or production settings are changed.
