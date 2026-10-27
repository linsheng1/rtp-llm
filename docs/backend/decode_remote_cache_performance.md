# Decode remote-cache performance validation

The delivery target is `alibaba/rtp-llm:feature/dsv41-dev-main-202161013`.
Refresh that branch before the final source freeze. Build, correctness,
monitoring and performance acceptance must identify the same committed source,
complete wheel, Python models and matching transport dependencies. Previous
wheels, native libraries or Python overlays do not qualify a new candidate.

## Feature-off compatibility

With `RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH` unset or `0`, the factory calls
the original SDK initialization APIs directly, including the original external
memory registration API when supplied. It does not inspect SSD backend JSON,
add Decode-specific backend restrictions, or allocate its SSD GPU packing arena.
This does not disable upstream Prefill caching or upstream GDR.

With the main switch set to `1`, the supported SSD path automatically selects
GPU packing from the `pace_ssd` backend. No additional packing enable switch is
introduced. Missing GPU registrations and unsupported mixed-backend layouts are
rejected before allocating a packing arena. CPU payload snapshots are not a
fallback.

P-only performance experiments that exercise this adapter explicitly enable
the main switch on P in both comparison arms. The private driver prevents
measurement-phase publication. Thus the comparison changes only the intended
read policy, rather than inadvertently changing the P transport path by using
different D-publication flags. Preparation writes remain separately reported.

Feature-boundary tests compare disabled factory behavior against the SDK's own
overloads, preserve the upstream non-dedicated-pool fallback when disabled,
and check enabled SSD validation. These checks passed in an intermediate
matching-SDK native run. Repeat the related tests on the final candidate;
native results do not establish feature-off model performance.

## Bounded read candidate

The supported SSD GPU factory supplies the originating device initializer.
Reads use windows of at most four objects and 256KiB of simultaneously live
payload. A single larger state/SWA object remains allowed by itself, up to the
existing 64MiB logical-operation limit. The shared registered GPU arena and
512MiB process accounting ceiling are unchanged; each operation charges its
largest live window. There is no read-window environment variable. The existing
single-object path remains for callers without a worker-device initializer.

Every concurrent object has a disjoint GPU span. All siblings are joined before
scatter, including after an SDK error, exception, device-initialization failure
or worker-launch exception. Uncertain operations retain the whole window until
SDK shutdown; no CPU payload fallback is introduced. Writes retain their
single-object completion ordering.

Native tests cover count/concurrency, device selection, disjoint data and
trace slices, byte admission, oversized state objects, device-init failure and
exception drain/quarantine. An intermediate matching-SDK snapshot passed these
native checks. Actual same-client SDK concurrency, error completion and buffer
lifetime still require independent runtime validation. Later telemetry and
safety changes must pass the final source's native and runtime gates.

## Fixed measurement protocol

Use the 32K by actual physical batch4 point first. Declare 21 alternating pairs
before collecting formal samples; retain all samples and every exclusion with
its reason. Compute paired absolute and relative changes and a 100000-resample
paired bootstrap confidence interval with seed 20261028. Report the median,
spread and individual pair changes. Do not select favorable pairs or expand
the sequence-length scan to resolve an uncertain result.

The serial control uses the same adapter, source, wheel, client and model as
the bounded-concurrency arm, with the delegate serialized by a private per-client
mutex. Call it the serialized-adapter control; it is not the historical SDK
implementation. Switch arms only after an idle barrier. Keep physical batch,
CP layout, block size, pool/reserve budgets, graph policy and cache state fixed.

Include the production reporter's callback path and actual inner SDK payload
observer in both arms. Record reporter/exporter configuration and its limits;
callback-only results do not establish exporter CPU overhead. Profile callback
cost, worker creation, D2D gather/scatter and each stream/device synchronization
separately from inner SDK time. Collect diagnostic traces separately from
formal samples. Avoid formal timing during same-host compilation or large IO.

Compare DRAM and recomputation controls under the same candidate and effective
workload. Record actual SSD/provider reads and warm-cache state; do not call a
page-cache read physical SSD throughput. Use private objects and namespaces,
without flushing shared caches or changing other users' services. Report the
remaining SSD/DRAM gap, transfer volume and the dominant residual costs.

## D publication timeline scope

P-only restoration measurements do not establish end-to-end D-publication
performance. The current common baseline fences the device with
`cudaDeviceSynchronize()` after EOS page admission in `planCompletedDecode`.
Measure that fence, whole-history admission, held GPU pages, queue delay and
EOS-to-commit separately from the actual D-only ranges written after
StartWrite masking. Do not replace the fence until an existing producer event
or execution-completion boundary demonstrably orders all FULL/scale/final
state/SWA writes, and coordinate that lifetime change with the functional owner.

## Historical candidates

The previous CP4 sharded, remote-SSD, 32K by actual physical batch4 experiment
used seven alternating pairs per candidate. Vector batching reduced the median
from 2146.15ms to 2093.78ms (2.44%). Bounded concurrency reduced its independently
measured median from 2215.28ms to 2049.33ms (7.49%). Each candidate used the same
wheel and model load in both arms; their samples are not pooled.

The concurrent candidate remains about 5.2 times slower than its 394.36ms
recomputation reference. All seven pairs improved, with two improvements below
3%; it does not meet the historical strong criterion that every pair exceeds
3%, and it does not establish session net gains. Overlap between the two raw
sample ranges is reported, but is not independently a reason to reject a paired
effect.
All four source histories' restored GPU bytes matched; restoration copy
observations were D2D, with no observed KV D2H/H2D. Those observations and the
384 native regressions apply only to the historical candidate artifacts.

Distinguish evidence of a positive paired effect from evidence that the effect
stably exceeds 3%. The fixed 21-pair protocol above replaces adaptive extension
for the new candidate. Estimates remain conditional on the recorded
model/Provider run and SSD cache state. P-only read gains do not prove session
net gains: include D publication cost and confirmed actual D-KV reuse.
