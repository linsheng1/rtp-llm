# Decode remote-cache performance validation

The common upstream baseline is `123290666d1439ebe842abf180a98c61230f2427`.
Build and native-regression acceptance for this baseline is separate from all
previous SSD candidate measurements. No previous wheel, native library or model
Python overlay qualifies the new baseline.

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

The new feature-boundary tests compare disabled factory error behavior against
the SDK's own overloads and check enabled SSD validation. They are added but
have not yet been executed on the new baseline. The existing
`ClientFactoryGdrTest.SsdAutomaticallyRequiresGpuRegistrations` in
`ClientWrapperTest.cc` also needs its main switch explicitly enabled; that
shared test adjustment is coordinated with the functional owner.

## New bounded read candidate (awaiting execution)

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

Five native tests cover count/concurrency, device selection, disjoint data and
trace slices, byte admission, oversized state objects, device-init failure and
exception drain/quarantine. These tests and this new upstream-based candidate
are not yet executed and do not inherit the historical results below.

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

For new candidates, report the paired absolute and relative differences, with
a fixed confidence-interval method declared before collecting further samples.
Distinguish evidence of a positive paired effect from evidence that the effect
stably exceeds 3%. If seven pairs indicate improvement but leave its magnitude
uncertain, extend only the same 32K by actual batch4 point using a predeclared
total pair count; keep every accepted and rejected sample with its reason.
Do not widen the long-sequence scan or select favorable pairs. These estimates
remain conditional on the recorded model/Provider run and SSD cache state.
