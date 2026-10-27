# Decode EOS publication through the Prefill remote-cache pipeline

Status: **review-only**, targeting `feature/dsv41-dev-main-202161013`.
No upstream or fork baseline has been merged. The current candidate generalizes
P CP / D DP configuration and supplies dedicated GPU memory registrations to a
matching PACE SDK. It retains native GPU pages after successful Decode completion;
there is no early CPU KV snapshot. Actual RDMA/GDR is a property of the matching
SDK, Consumer, Provider and transport configuration, and must be verified at both
ends rather than inferred from an environment variable.

The current functional and capacity qualification is maintained separately from
the historical release-port results below. Native GPU/mock tests do not qualify
physical CP8/CP16 model execution. Large synthetic backend payloads do not qualify
large model KV histories or net session performance.

## Lifecycle and checkpoint selection

1. `GenerateStream` submits once at successful completion, before the final block
   remap; `StreamCacheResource` provides an idempotent release-time fallback.
   Cancelled/failed requests and warmup requests without a cache manager do not
   publish. The supported path is single-token autoregressive Decode, one beam,
   with cache reuse enabled and asynchronous stream reservation disabled.
2. The computed KV endpoint is `seqLength() - 1`: the last sampled token has not
   been evaluated by the model. There is no per-token CPU payload capture or
   additional per-token device fence.
3. Select an actually resident complete checkpoint on the common boundary of
   physical CP sharding and reusable state/SWA page spans. Validate every FULL
   page and the required terminal state/SWA pages. If a page is absent, back off
   to an earlier complete checkpoint; never relabel partial final state with an
   earlier key. Leave the unpublished tail for the next Prefill to recompute.
4. For replicated CP, write the same full blocks to each target P rank's baseline
   spec. For sharded CP, use `CPSlotMapper` for round-robin FULL pages and native
   state/SWA slices. Target P rank count and physical shard count are distinct.
5. `StorageBackend::prepareWrite()` pins all unique source pages, including CP
   peer pages, before request ownership is released. Reuse the existing executor,
   Prefill `StartWrite`, key mask, tag batches, actual-URI handling and
   `FinishWrite` transaction. Local hot-cache insertion remains enabled.
6. Pins remain until transfer completion. If SDK completion is uncertain, preserve
   the release branch's permanent `FROZEN` references until block-pool destruction.
   Retain the EOS admission lease until backend shutdown so frozen pages continue
   to count against publication admission while the service is accepting work.
   SDK shutdown does **not** unfreeze these pages.
   A lost commit acknowledgement must not abort a possibly committed
   transaction or indefinitely retain payload whose transfer already completed.

One EOS device fence makes the final GPU writes visible before asynchronous
publication. FULL pages and final state/SWA data are sent through the existing
Prefill transaction and GPU IOV path. Dedicated GPU registration is mandatory
when GDR is requested; an unsuitable host-pinned or non-dedicated pool fails
initialization rather than silently switching to CPU payload staging.

## General configuration

Use the same feature setting and target layout on P and D:

```sh
export RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH=1
export RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY=1
```

`prefill_cp_size` is the target P CP count on both roles. P uses TP equal to
that count and DP1; D uses TP1 and an independent positive `dp_size`. For example,
P CP2 / D DP4 and P CP8 / D DP4 use exactly the same feature switch. There is no
CP2/CP4 whitelist, instance-group name prefix requirement, or DP=CP requirement.
The model's own shape, expert-parallel and hardware constraints still apply.

Select replication with the existing `prefill_cp_kv_cache_sharded=0`, or sharding
with `prefill_cp_kv_cache_sharded=1`. Both roles must agree. Cache identity includes
target CP, physical shard count and base block size automatically. Do not reuse
old-format remote keys across layouts; the identity change deliberately causes a
cold miss rather than a potentially incorrect cross-layout hit.

For DRAM select the existing `pace` SDK backend and read backend 3. For explicit
SSD select `pace_ssd` and read backend 9. GPU packing is selected automatically. Backend selection is independent of Decode
publication, so ordinary Prefill GDR does not require a second experimental PACE
flag. The old CPU packing/coalescing implementations have been removed from this candidate.
SDK calls are always bounded by byte, block and IOV limits; individual SSD objects
remain intact. Trace block IDs are sliced along with each SDK batch.

The transport configuration must select RDMA without a TCP payload fallback.
Configure the matching Consumer's GDR read/write and external-memory flags,
`MC_DISABLE_GPU_DIRECT_RDMA=0`, and the real local rank count for
`TAIR_MEMPOOL_EXPECTED_EXTERNAL_CLIENTS` (the actual local client-process count
for that Consumer: CP/DP count in the single-host-per-role examples; per-node local
rank count for multi-host roles). A Provider does not inherit the Consumer's rank count. GPU
registration success alone does not prove that a closed-source Consumer avoided
D2H/H2D: validate registered payload address ranges and Consumer CUDA activities
with a working same-process positive control.

There are no legacy CP-specific or duplicate publication enable flags. The
single publication switch defaults to off. Backend and CP layout selection are
independent existing configuration, not additional Decode feature flags.

### Initial resource budgets

Per D rank, EOS publication charges at most **512 MiB** of GPU source pages
across at most **4** queued/in-flight jobs. Pages are deduplicated within each
job; shared pages across separate jobs are conservatively charged again.
These are admission budgets,
not a second preallocated KV pool. Against the present 4 GiB/rank test pool the
byte budget is 12.5%; this is a deployment starting point, not a measured optimum.
Advanced overrides are `RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB` (1–16384) and
`RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS` (1–64), strictly parsed once at backend
initialization. Refusal skips optional remote publication, retaining the normal
response/local cache. Uncertain transfers keep admission charged through FROZEN
references. Existing shared queue limits and the P/D quarantine mechanism remain.
`RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO=0` remains the default: disabling
threshold-triggered process abort does not disable page freezing.

Payload checksum diagnostics are removed from business execution. Tests use
external GPU/Consumer observers and byte comparisons, with their diagnostic
copies kept outside the measured KV transfer interval.

The enabled path currently rejects MTP/speculative decoding and asynchronous
stream reservation explicitly. It requires a native single PACE backend, valid
base-block geometry and a nonempty instance salt. Normal feature-off behavior
remains available. Request errors/cancellation do not publish. Deployment-level
cache-tier policy and the existing request reuse-cache/override policy retain
their ordinary meaning. With publication enabled, D remote lookup is intentionally
local-only; D receives its prompt KV through the existing P-to-D mechanism and
publishes completed additions for P. This does not add remote-to-D reconstruction
from the P-layout wire format. D local hot-cache reuse remains enabled.

## Admission and failure isolation

No GPU shadow KV pool or request-owned early CPU payload snapshot is allocated.
The cost is delayed reclamation of existing source GPU pages. Local hot-cache
insertion and ordinary Prefill remote publication remain enabled.

| Setting | Default | Meaning |
| --- | --- | --- |
| `RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH` | `0` | One opt-in switch for all supported P CP / independent D DP layouts. |
| `RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB` | `512` | Per-D-process retained source-page charge, deduplicated per job and including quarantine; range 1–16384 MiB. |
| `RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS` | `4` | Per-D-process publication jobs, including quarantine; range 1–64. |

Admission accounts for the entire candidate checkpoint before the metadata mask
is available; masked pages are not released early. Exhaustion skips optional
remote publication while preserving the response and normal local cache. The
retained-page limit is not an extra HBM allocation or a proof of zero capacity
loss.

The SSD adapter uses one shared registered 64 MiB GPU arena per process/device,
with a 512 MiB per-process registered-arena cap. Operations lease disjoint ranges
sized to their largest URI, not their entire logical batch. GPU writes gather one
URI immediately before its SDK write; reads scatter its result before the next
URI overwrites that range. This avoids starving concurrent cache tags when a
replicated SWA batch approaches 64 MiB. The dedicated GPU copy callback explicitly waits for its
default-stream D2D copy before returning, because external RDMA is not ordered
by CUDA stream dependencies. This is a per-packing-copy completion guarantee,
not a new per-token inference fence.
Unknown SDK completion quarantines only that operation's range; independent tags
can continue using other ranges. The affected client fails closed until shutdown,
and its delegate drains before its quarantined memory is released. This protects
pending DMA and can reduce available capacity; it does not make uncertain writes
successful. The limits are transient-operation limits, not remote DRAM/SSD pool
capacity limits. SSD uses D2D gather/scatter, not a CPU KV snapshot; this does not
claim GPUDirect Storage on the remote SSD host.

## Validation and limits

### Historical v5 validation (before configuration cleanup)

Native GPU/mock coverage is CP1/2/4/8/16 × sharded/replicated × independent
DP1/3/8. The historical v5 baseline's 377 executed tests passed with zero skips, including GPU
packing/scatter ordering, concurrent large logical batches and unknown-completion
range isolation. Real-model qualification includes CP2 P / DP4 D and CP4 P / DP4 D,
with both layouts and DRAM/explicit SSD. Execution results, exact wheel/source
hashes, actual pool budgets and Consumer observations must be taken from the
matching test report. Unexecuted combinations are not passes.

The current large backend test generates distinct 32 MiB objects entirely on GPU,
commits the actual writes, and uses an independent reader process to compare every
64-bit word on GPU. The only readback is an 8-byte diagnostic mismatch count per
object, outside the registered payload range. Target payloads are 128 GiB in a
160 GiB DRAM pool and 512 GiB in a 640 GiB SSD backing file. Actual SSD allocation,
process IO and O_DIRECT descriptors are recorded; sparse file length alone is not
acceptance. These byte tests and the real-model tests are reported separately.

### Current executed result (generalized v5)

Full wheel SHA256: `628b4aa3c8c0cdaa29f49d28e2dbff1054a919c396edb2c4164b9136ce04eee2`;
source overlay SHA256: `e8d5831fff2c5373bdfb9596e14c0008e4b2a4279f105d05fa721348716cc2bd`.
All 377 native tests executed with zero failures or skips. All four actual-model
paths below passed EOS publication, new-P cold recovery, local-hot/bypass exact
token comparison, and normal model/Consumer shutdown:

| Backend | Layout | P CP | D DP | Published endpoints | Result |
| --- | --- | --- | --- | --- | --- |
| DRAM | sharded | 2 | 4 | 8192, 8192, 8192, 8448 | Passed |
| DRAM | replicated | 4 | 4 | 8192, 8192, 8320, 8448 | Passed |
| SSD | replicated | 2 | 4 | 8192, 8192, 8320, 8448 | Passed |
| SSD | sharded | 4 | 4 | 8192, 8192, 8192, 8192 | Passed |

Both model and Consumer observations show zero KV-payload D2H/H2D in the
qualified route. SSD GPU D2D packing is counted separately. Actual SSD cold
reads were 19,292,160 and 9,449,472 bytes, with O_DIRECT descriptors observed.
The model pools were 4 GiB/card; this is correctness qualification, not a maximum
HBM-capacity or performance-frontier result. CP8/CP16 and DP3/DP8 have native
GPU/mock coverage, without physical multi-node model qualification.

Large backend verification separately passed 128 GiB DRAM and 512 GiB SSD
full-word GPU checks, using the v3 single-URI probe. Its source revision and
copy diagnostics remain distinct from final v5 model evidence. In v3, the last
CP2-replicated model attempt revealed ordinary P write failures despite matching
continuations. V4 switched GPU packing leases from logical-batch size to the
largest URI; v5 explicitly completed D2D copies before external RDMA handoff.
Failed and interrupted attempts are retained and not counted as final passes.

### Historical qualification (different revisions)

### Initial release-port attempt (historical)

Code under test: `badca4c96d70c0639afbb05ea66c1be9c87c4f0f`;
feature implementation `8b9ee641`. The historical documentation-only update `565ca78` did not change that
initial tested implementation. Subsequent source fixes are qualified separately below.

| Layer | Result | Boundary |
| --- | --- | --- |
| Standalone C++17 coalescer | 79 assertions; ASan/UBSan passed | Public API headers + mock SDK; CPU only. |
| Fresh ARM64 native build | Passed | Exact release-port tracked source; no old library substitution. |
| `native_dtop_completion` | 28/28 | Real GPU pages + mock SDK, CP2/CP4 sharded/replicated. |
| `client_wrapper` | 22/22 | No skips. |
| `kvcm_config` | 9/9 | No skips. |
| `kvcm_local` | 15/15 | Test-only baseline corrections below; no skips. |
| `storage_backend` | 36/36 | No skips. |
| GPU coalescer | 132 assertions | Real GPU allocations + mock delegate. |
| Full ARM64 wheel and two-host loader | Passed with dependency supplement | Stock public target still lacks DeepGEMM. |
| Real-model DRAM/SSD × CP layouts | **Failed overall: first DRAM/sharded functional cases passed, shutdown crashed; other three paths not run** | Conditional runtime dependency fix described below. |

The first KVCMLocal run's three failures reproduced on the exact upstream
`a06c91abe` baseline. The test-only correction uses the observed failed-init
shutdown contract, checks unchanged quarantine generation/task/block counters
after a metrics snapshot consumes its samples, and scopes the CUDA death test
to GTest's threadsafe re-execution mode. It still requires a real SIGABRT for
the abort-threshold case. The adjacent invalid-topology expectation remains
zero. Original failed attempts were preserved; these fixes do not change
production behavior.

The stock public complete-wheel target fails dependency analysis because its
CUDA13 lock omits `@pip_cuda13_arm_torch//deep_gemm`; the requirements explicitly
expect an internal overlay. The test environment supplies a separate, pinned
DeepGEMM repository: `2.8.2+ad62a90.cu132`, SHA256
`2e0e352bd2a07be8f6e3eb42edac0b57b423e8ddaaaffff1057ecb11438d20f9`.
This is **not** an official internal overlay or a stock-public-build pass.
Product wheel SHA256:
`9f65dc24a0f4f169dca7ace9d986faf6b30355aa603b4f302e697f7a388ab3a5`.

The first real P startup also reproduced the locked FA4 `4.0.0b21` / Quack
`0.6.5` missing `quack.activation.sub_packed_f32x2` error. The loaded package
paths and versions were verified. The independent second test runtime uses
[upstream FlashAttention #2787](https://github.com/Dao-AILab/flash-attention/pull/2787)
(commit `145b1010051dbfd4bdc41a0ae55d495b08d7a458`), replacing three call sites
with the CuTe primitive and removing obsolete imports. Both hosts' 1112
installed RTP files remain identical to the product wheel. Two SM100 BF16
forward cases passed an FP32 SDPA reference (max absolute errors 0.00927 and
0.00905). No shared runtime was changed. This dependency fix is an explicit
acceptance condition, not a change silently included in the RTP PR.

The real model test used CP4 P, DP4 D, and a separate RDMA-staging DRAM
Provider, with 36 GiB/card KV pools and local-hot caching enabled. Four independent
7680-token inputs produced actual outputs of 513/530/641/777 tokens. Their
computed endpoints were 8192/8209/8320/8456; the complete sharded checkpoint was
8192 in all four cases (tails 0/17/128/264). After restarting P, local reuse was
zero and remote reuse was 8192; hot-local reuse was 8192 and bypass reuse zero.
All four cold/hot/bypass continuations matched for all eight actual tokens.
These 512 additional KV positions include the first P-sampled output token
whose KV had not yet been computed by P; do not call all 512 pure D-sampled tokens.

**The run failed shutdown acceptance.** D rank 2 stopped its engine and destroyed
the model while peers were still executing `NormalEngine::step()`. It then
SIGSEGVed after BackendManager shutdown. Peers timed out in DeepGEMM grid sync
and SIGABRTed. The D service root returned zero, but native logs and nonzero
reaped helper exits expose the failure. During subsequent failure cleanup P
exceeded the 600-second shutdown bound and required an identity-checked SIGKILL
of its own private PID namespace; that is not a separate isolated P-only normal
shutdown reproduction. All test model/Consumer/Provider processes and the private
CPU weight keeper were subsequently released.

In that initial attempt, `NormalEngine.cc` was byte-identical to the release baseline. The older
experimental branch's per-step collective stop-vote workaround was not ported.
The observed ordering is a strong investigation lead, not a symbolized root cause
for the first SIGSEGV or proof of EOS-publication causality. At that point no B-disabled model
counterfactual had been run. The follow-up below supplies that evidence and adds
an explicit shutdown-only RPC fence.
DRAM/replicated and both explicit-SSD model configurations were not started after
this blocker; native CP/layout coverage does not qualify those real model paths.

Official release dependency-overlay CI remains required before merge. This
port does not establish GDR or maximum stable capacity/performance results.
Native tests use a mock SDK; real-model token comparisons do not establish
full-logit equivalence. Optional read fingerprints add diagnostic D2H, and the
new EOS path does not emit a paired source fingerprint for every actual SDK
write. Therefore no complete source/restore byte-fingerprint match is claimed.
Active model cancellation and real-SDK error injection are not covered by the
normal-EOF matrix; native failure/ownership tests do not replace those runs.
Explicit SSD tests exclude automatic DRAM-to-SSD tiering and do not force MR
cache eviction or establish physical SSD reads on every restore.

### Shutdown follow-up on the same release baseline

The B-disabled, idle DP4 Decode counterfactual also reproduced the first
SIGSEGV. GDB and symbolization identify the KMonitor deleter in
`MetricsReporter.cpp:78`: the process-wide grammar backend outlived
`KMonitorFactory::Shutdown()`. Its metrics reporter then dereferenced a deleted
KMonitor. The relevant baseline grammar/factory/metrics sources match a06;
a complete pure-a06 model counterfactual was not run.

The first fix released the grammar backend before the factory and coordinated
engine-loop exit over RPC. A subsequent real DRAM/sharded run completed its
cache data comparisons and same-step loop fence, but the first P still aborted
in the ANet metrics thread. Its root returned zero while rank 0 exited by
SIGABRT. Python attention metadata retained an async writer, whose allocator
and active-store references kept the transport/reporting threads alive.

The candidate now also explicitly shuts down the writer at native model
teardown, joins submitted tasks, makes late completion callbacks terminal, and
releases both the allocator and its aliasing configuration pointer. Python
aliases may survive, but cannot reopen the closed writer. Source-page leases
owned by outstanding underlying store callbacks keep their existing lifetime
contract.

The loop fence tracks completed/in-flight iterations on the CPU; shutdown RPCs
advance lagging ranks to a common completed boundary and distribute an
incarnation-bound certificate before peers close their RPC servers. There is
no per-step GPU collective stop vote. CPU mutex/counter work is added to the
normal loop; its isolated real-model performance impact is not measured here.

Candidate source overlay SHA256:
`e94d84cdae3108e5cf89e21e0731ce26a6d7b6705f9f658a951d749bd69e8407`;
complete wheel SHA256:
`61861e2a6b4b479ca5e2c20a64168026121085f3a1be522b757b306694c41518`.
Fresh native results: 12 shutdown/loop coordinator + 3 real grammar/KMonitor
ownership + 52 async-writer cases, all passed with zero skips. The initial
110 EOS/backend cases and earlier 8 dispatcher-stop cases remain separately
versioned evidence. The candidate product sources were committed as `29b6b66` (owner teardown) and
`6bedc63` (same-step shutdown fence), byte-identical to the tested overlay.
The four-path real-model matrix passed:

| Backend | CP layout | Published endpoints (four histories) | Data and full lifecycle |
| --- | --- | --- | --- |
| DRAM | sharded | 8192, 8192, 8192, 8192 | Passed |
| DRAM | replicated | 8192, 8192, 8320, 8448 | Passed |
| SSD | sharded | 8192, 8192, 8192, 8192 | Passed |
| SSD | replicated | 8192, 8192, 8320, 8448 | Passed |

All 16 histories passed new-P cold/local-hot/bypass exact token comparisons;
followups permit natural EOS and compare all actual outputs up to eight tokens.
All 48 native rank exits were independently observed as exit code 0, with equal
completed-step boundaries within each incarnation. No SIGSEGV/SIGABRT, forced
termination or surviving own model/Consumer/Provider process was accepted.
Original token outputs and checksums were copied to the local evidence bundle.
The private CPU model-table keeper was released after the final audit.

These are functional and normal-shutdown results under the dependency and
explicit-SSD conditions above. Active cancellation, real-SDK error injection,
GDR, physical SSD-read enforcement, automatic-tier CRC, capacity and performance
are outside this follow-up. Official release-overlay CI remains required; the
PR stays Draft for human review and has not been merged.

### Historical evidence from the original experimental branch

The original implementation revision was built into a complete ARM64 wheel and
tested on GB200 before this release port. Historical wheel SHA256:
`c7a8867cc6ab56c492ba2e9785ed1d89aa2ef02519dc49183194c1383dc2a762`.
These results do **not** qualify the new release-branch port.

The historical native GPU test suite passed **28/28**, with a mock SDK. It covers CP2/CP4
and sharded/replicated layouts, source lifetime after request release, final
state/SWA bytes, partial tails, missing state, non-reusable SWA, duplicate EOS,
admission rejection, unknown SDK completion and lost commit acknowledgement.
Relevant targets (append the environment's CUDA/remote-cache build configuration):

```sh
bazel test //rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:native_dtop_completion_test
bazel test //rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:client_wrapper_test
bazel run //rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:coalesced_pace_write_test_bin
# Optional real-GPU adapter checks:
bazel run //rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:coalesced_pace_write_test_bin -- gpu
```

The real-model check used CP4 P, DP4 D and a separate remote host, with the same
36 GiB/card KV pool and D admission settings for both layouts. Each of four
DRAM/direct-SSD × sharded/replicated configurations generated four independent
7680-token prompts with actual Decode output lengths of 513, 530, 641 and 777.
A newly started P, local-hot reuse and bypass were checked separately.

| Layout | Computed endpoints | Published endpoints | Tail recomputed by P |
| --- | --- | --- | --- |
| Sharded CP4 | 8192 / 8209 / 8320 / 8456 | 8192 / 8192 / 8192 / 8192 | 0 / 17 / 128 / 264 |
| Replicated CP4 | 8192 / 8209 / 8320 / 8456 | 8192 / 8192 / 8320 / 8448 | 0 / 17 / 0 / 8 |

All 16 fresh-P remote hits equalled the published endpoint, with local reuse
zero; hot-local hits matched it and bypass reuse was zero. All 16 cold/hot
continuations matched. Strict three-way equality over the full actual output
held for 12/16: DRAM naturally ended after two tokens; direct-SSD replicated
matched eight tokens without EOS. Direct-SSD sharded was forced to eight tokens
despite natural EOS at token two: bypass diverged **after EOS** in four cases.
Those cases are not counted as strict eight-token passes, and no SDK corruption
or floating-point root cause is asserted. All tested P/D/Consumer processes
drained normally, with no SIGSEGV/SIGABRT or forced termination.

The direct-SSD check used an 8 GiB Provider pool, a 10% MR cache and a 256 GiB
backing file. It did not force MR eviction and does not prove every restore read
physical SSD. It does not cover the older automatic DRAM-to-SSD tiering CRC issue.
Long-output/full-logit equivalence, performance gain and maximum stable KV pool
capacity have not been established for this EOS-retention implementation.
