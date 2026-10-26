# Decode EOS publication through the Prefill remote-cache pipeline

Status: experimental extension of the remote-gdr branch. This implementation
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
6. Pins remain until transfer completion. If SDK completion is uncertain, retain
   them through the existing quarantine until SDK shutdown drains outstanding
   access. A lost commit acknowledgement must not abort a possibly committed
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

The existing branch's narrow experiment gate remains: `RTP_DTOP_PACE=true` and
`RTP_DTOP_FILE_CP2=true` (the inherited gate name also covers CP4),
matching `RTP_DTOP_FILE_TARGET_P_TP=2` or `4`, 128-token base blocks, no MTP or
speculative decoding, P CP2/CP4 with TP matching CP, D TP1 with DP2/DP4 (EP4 for
DP4), an explicit single PACE backend, nonempty instance salt, and an isolated
instance group beginning with `linshuo_decode_remote_`. Replication is selected
by the existing `prefill_cp_config.kv_cache_sharded` setting, not a second pool.
These guards are inherited experimental constraints, not a general deployment API.

## Validation and limits

The implementation revision was built into a complete ARM64 wheel and tested on
GB200. Wheel SHA256:
`c7a8867cc6ab56c492ba2e9785ed1d89aa2ef02519dc49183194c1383dc2a762`.
Subsequent PR cleanup changes formatting, comments and documentation, removes a
standalone probe, and marks the coalescer executable as test-only; it does not
constitute another GPU/model test run.

The native GPU test suite passed **28/28**, with a mock SDK. It covers CP2/CP4
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
