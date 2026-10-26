# Decode EOS publication through the Prefill remote-cache pipeline

Status: **review-only port**, based on the upstream
`feature/dsv41-dev-main-202161013` snapshot
`a06c91abea437bf85cd5bc6634b0e142093ebe2e`, copied unchanged into the fork.
Changes live on a separate branch; neither upstream nor the fork baseline has
been merged or modified. ARM64 native regression on this base passed;
full-wheel and model acceptance conditions are recorded below. The PR remains Draft.
This implementation
retains native GPU pages after successful Decode completion. It does **not**
implement SDK/Consumer GPU registration or establish GDR support.

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
publication. GPU IOVs reach the transfer client, but the current SDK can still use
CPU staging internally. Native page retention is not proof of zero-copy transfer.

## Admission and opt-in settings

No GPU shadow KV pool or early request-owned CPU snapshot is allocated. The cost
is delayed reclamation of existing GPU KV pages. Admission currently accounts for
the entire candidate checkpoint before the metadata mask is available; masked
pages are not released early.

| Setting | Default | Meaning |
| --- | --- | --- |
| `RTP_DTOP_GPU_PENDING_MIB` | `4096` | Per-D-process unique retained source-page budget, including quarantine; accepted range 1–16384 MiB. |
| `RTP_DTOP_GPU_PENDING_JOBS` | `2` | Per-D-process publication job budget, including quarantine; accepted range 1–64. |
| `RTP_DTOP_DECODE_PUBLISH` | `true` | Enable EOS publication after the experiment has been explicitly configured. Set false for a Prefill-only control. |
| `RTP_KVCM_BOUNDED_SDK_BATCH` | `false` | Split SDK calls at 64 MiB, 256 blocks or 1024 IOVs while preserving trace metadata and actual-URI order. |
| `RTP_KVCM_COALESCE_PACE_WRITES` | `false` | Pack one URI's IOVs into a bounded CPU staging buffer for the explicit PACE DRAM backend. |
| `RTP_KVCM_COALESCE_2D_PACK` | `false` | With coalescing enabled, combine verified regular GPU rows with `cudaMemcpy2D`. |

When admission is exhausted, skip the optional remote publication and preserve
the generated response and local cache. The retained-page budget is not an extra
HBM allocation. Coalescing separately limits staging to 64 MiB per call and
512 MiB per process. It is mutually exclusive with the packed SSD adapter and is
explicitly a CPU-staging optimization.

The original implementation's narrow experiment gate remains: `RTP_DTOP_PACE=true` and
`RTP_DTOP_FILE_CP2=true` (the inherited gate name also covers CP4),
matching `RTP_DTOP_FILE_TARGET_P_TP=2` or `4`, 128-token base blocks, no MTP or
speculative decoding, P CP2/CP4 with TP matching CP, D TP1 with DP2/DP4 (EP4 for
DP4), an explicit single PACE backend, nonempty instance salt, and an isolated
instance group beginning with `linshuo_decode_remote_`. Replication is selected
by the existing `prefill_cp_config.kv_cache_sharded` setting, not a second pool.
These guards are inherited experimental constraints, not a general deployment API.

## Validation and limits

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
