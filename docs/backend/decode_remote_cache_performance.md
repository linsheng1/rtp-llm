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

## Historical candidates

The previous CP4 sharded, remote-SSD, 32K by actual physical batch4 experiment
used seven alternating pairs per candidate. Vector batching reduced the median
from 2146.15ms to 2093.78ms (2.44%). Bounded concurrency reduced its independently
measured median from 2215.28ms to 2049.33ms (7.49%). Each candidate used the same
wheel and model load in both arms; their samples are not pooled.

The concurrent candidate remains about 5.2 times slower than its 394.36ms
recomputation reference, with overlapping ranges and two paired improvements
below 3%. It does not establish stable performance gains or session net gains.
All four source histories' restored GPU bytes matched; restoration copy
observations were D2D, with no observed KV D2H/H2D. Those observations and the
384 native regressions apply only to the historical candidate artifacts.
