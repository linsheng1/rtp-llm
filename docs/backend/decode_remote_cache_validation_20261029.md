# Decode remote-cache validation, 2026-10-29

Status: the finite SSD writer comparison is complete, but the final candidate's
strict cold-versus-bypass output gate failed. This is an evidence record, not
an unconditional runtime acceptance certificate. A minimal original-source
counterfactual for the same new input is still pending; upstream delivery has
not been performed.

The target is `alibaba/rtp-llm:feature/dsv41-dev-main-202161013` (verified head
`c68265783f75b2a638fadd2c422d4c9be11a47bc`). Delivery preserves existing history
and adds commits. Functional runs use the existing matching SDK1025 and Manager;
the repository's original SDK dependency stays unchanged. SDK source changes or
replacement SDK binaries are not part of this delivery.

## Artifact and test identity

The completed long model runs use product commit
`5daf1461cd7db09e6abb894e61f169f527fcff6f`, tree
`115513ab2fa7a0adc21a5a6bab4e1c87cbfd6d46`.

| Environment | Complete wheel SHA256 | Executed native cases |
|---|---|---|
| Old GB200, DRAM | `512ca8f6881e4afe9721b481f551cbaa81d7df472e76a017c100ca3c0332a963` | 487 passed across eight targets |
| Second cluster, SSD | `8a4458ed3193119ac125304a25b183c7bf8c574396eb15c1443143a9ea8a9bb4` | 395 passed, including the subsequent test-only local-cache change |

The SSD installation verified all 2066 product members of the complete wheel.
SDK1025's source RPM SHA256 is
`5b382202aa716a850984ce2acfb03981b05875b5b014cc8e9bc215dee22ddae4`;
the normally packaged SDK DSO SHA256 is
`eccbafbcdc651c65df5c029f8f3326e954f1dbe601ed651b27c7fc69562d25ed`.
The second cluster's private RPM repack preserved all five original headers and
the DSO byte for byte; its container hash is recorded separately.

The subsequent compatibility change `a16022d` preserves original SDK backend
metadata responses when Decode publication is unset or disabled. Its exact
SDK1025 wrapper binary passed all 31 cases separately in unset, zero and one
ambient switch modes. With original SDK794, the three explicit feature-off
cases passed in each of those modes. A broader CPU-only run retained its
30/31 result: the GPU-registration factory test cannot reach its expected throw
without a visible CUDA device. It is not counted as a full passing GPU run.

SSD write candidate `ff13c95c5e6ce44cddad6789c4eb6acf2758a2e8` adds bounded
64-object/2MiB windows on top of a16022d. Its exact patch SHA256 is
`0ab5c95ea33e6268066452684d896ae34dfc3e8d00f5f2ea2fce92378b8f99e9`.
Its 297 NativeDtop GPU/mock-SDK cases and 31 wrapper cases passed with no
failure, error or skip. The initial dangling-else test compilation failure was
retained; the second patch only added the necessary test braces.

Final ff13 product-header compatibility was also compiled against original
SDK794 using the existing CPU-only native target: build exit0, 10.215s and
four actions. The three explicit feature-off cases each passed in unset, zero
and one ambient switch modes, nine executions total; all three XMLs contain
three completed cases and no failure, error or skip. This does not claim a new
SDK794 GPU test or a full wheel. A first attempt exited36 before build actions
because the private process lacked USER; setting the actual account's
USER/LOGNAME fixed only that launch environment, and the failure is retained.

Scope matters: DRAM model, copy and formal performance evidence remains bound
to its complete 5daf wheel. The later SSD write implementation and feature-off
metadata delta have their own exact-source gates; the 5daf DRAM wheel is not
relabeled as a full ff13 DRAM model run.

A standalone unchanged-SDK1025 probe issued one Save for 64 independent 32KiB
GPU objects, committed, re-queried URIs and loaded in bounded read windows.
All 2MiB compared equal, cache removal and client/Consumer teardown exited zero.
This is SDK capability evidence, not model-KV or performance acceptance.
The candidate whole-wheel Bazel build exited zero. All 2070 archive members
were verified; its 1,943,499,127-byte wheel SHA256 is
`a16c2c9808209861636e519d7bb1d722d61d9965cad8f8f926dd665d47313d39`.
The main library SHA256 is
`88a3f2aa4a24edd71dca09d6f67b04bac6da9dd8de44c5f792b1feeea247f91a`;
the compute library and SDK DSO are unchanged. Both P and D installed all 2066
product members. Fresh CPU processes using the official
`ensure_engine_ops_loaded()` entry point exited zero without initializing CUDA;
all three mapped product/SDK libraries came from the new wheel directory and
their hashes matched. Integrated model results are still pending. An initial
probe incorrectly expected engine mappings after a lazy import and exited1;
that failure remains intact, with independent successful CPU gate records.
A post-build whole-tree assertion failed
after test assets materialized; that failure is kept separately from the
successful build and product-source verification. Verification reused the same
build output without recompiling.

## Real Decode KV and fresh Prefill restoration

| Medium / layout | Real source | D committed | First fresh-P input and reuse | Output comparison |
|---|---|---|---|---|
| DRAM, CP4 sharded | 1536 input + 513 output | 2048, 1 location, 1,976,320B | 2177 input, local0/remote2048 | Both 32 tokens; first cached/bypass difference at index7 |
| SSD, CP4 sharded | 7680 input + 1025 output | 8704, 2 locations, 2,484,224B | 8833 input, local0/remote8704 | Both 32 tokens; first difference at index7 |
| DRAM, CP4 replicated | 1536 input + 513 output | 2048, 4 locations, 7,806,976B | 2177 input, local0/remote2048 | Both 32 tokens; IDs differ |
| SSD, CP4 replicated | 1536 input + 513 output | 2048, 4 locations, 7,806,976B | 2177 input, local0/remote2048 | All 32 IDs equal |

Real token IDs, computed and aligned published endpoints were retained. A
1536-to-2048 restore covers 512 D-computed KV positions beyond the original P
boundary. If the first sampled output came from P, the corresponding count of
D-sampled tokens with computed KV is 511. The final sampled token is not assumed
to already have computed KV. These are controlled source-boundary observations;
production D-origin metadata and dedicated metrics remain unaccepted end to end.

The sharded cached/bypass discrepancies are retained. A P-only checkpoint
control also differed, even with Decode publication disabled. An independent
unaltered official `9115e23` whole wheel and original SDK794, with remote cache
disabled and no Consumer, reproduced both complete 32-token output sequences
exactly using local prefix reuse versus bypass. This establishes a pre-existing
P-checkpoint numerical difference. It is not a pure upstream remote test, a
GPU-byte comparison, or a blanket correctness exemption for D-produced KV.
The original SDK794 and matching SDK1025 have incompatible metadata ABI layouts;
no mixed-SDK model experiment was performed.

## Single four-session, two-turn comparison

Each medium ran one A/B comparison: four concurrent sessions per arm, each with
two turns. Every R1 generated exactly 32768 actual tokens. A disables only D
publication; B enables it. P reader settings are identical. Corresponding
session histories and R2 inputs matched across arms. R2 followed all four EOFs
without a publication wait, with an observed scheduling gap of 49–53ms.

| Medium / arm | R2 TTFT ms, session0..3 | Median ms | Range ms |
|---|---|---|---|
| DRAM A | 805.564, 507.085, 354.770, 657.251 | 582.168 | 354.770–805.564 |
| DRAM B | 286.698, 723.414, 575.758, 431.858 | 503.808 | 286.698–723.414 |
| SSD A | 732.245, 578.810, 422.987, 273.009 | 500.898 | 273.009–732.245 |
| SSD B | 337.226, 484.863, 188.354, 632.290 | 411.044 | 188.354–632.290 |

All 16 requests per medium reached normal EOF, but every R2 reused only local7680
and remote0. These TTFT differences do **not** demonstrate D-cache gains. No P95,
statistical significance or cross-hardware DRAM/SSD medium comparison is claimed.
D commits lagged EOS by about 2.5s on DRAM and 7.56–8.73s on SSD, after R2 lookup.

For SSD, each R1 publication made four outer chunks of 191 URI objects: 764
singleton inner Saves per transaction, 3056 for four transactions. Outer SDK time
was approximately 7.54–8.71s; Manager StartWrite was about 9ms and FinishWrite
about 0.9ms. Individual inner-call latency was not instrumented. This motivates
bounded batching but does not itself quantify a speedup. Background writes and
Manager acknowledgements after RPC EOF are included in separate ledgers; missing
legacy feature-off wrapper logs are not treated as zero I/O.

The DRAM decomposition points elsewhere: D-local EOS-to-first-SDK took
2.094–2.272s, the four-call SDK envelope 28.7–30.9ms, and last-SDK-to-commit
337.9–385.2ms. Manager-local StartWrite took 1.708–1.911s and FinishWrite
0.308–0.349ms. These independently timed stages implicate control-plane
waiting; cross-host clocks do not justify subtracting them into an exact
per-rank residual. The DRAM logs do not support treating singleton SDK writes
as its main bottleneck, and no unmeasured Manager change is included.

The SSD controller incorrectly waited for an actual endpoint in a legacy field
that reports the aligned published endpoint after all requests had completed.
Its 240s post-processing timeout remains a harness failure, separate from the
16 successful requests. It was not hidden by rerunning the long cohort.

## Copy observations, timeline and shutdown limits

Controlled GPU registrations, remote read/write observation windows and
Consumer bidirectional positive controls were captured on both media. Within
the covered SDK windows no registered KV H2D/D2H was observed; GPU packing used
D2D. Graph and GetProcAddress coverage is incomplete, so this does not establish
zero CPU staging across the entire service. SSD replicated P/D handoff had
registered KV D2H after the SDK read window; it is recorded separately.

Actual rank0 P-only diagnostic timeline JSONs were saved and parsed: DRAM
30,612,307B and SSD 30,735,856B. Both start after remote restoration, cover a
single diagnostic request/capture using a later 40448-token checkpoint plus128 tail,
and are not the formal four-request R2 timeline. They cannot measure SDK restore
latency. An initial SSD trace was missing despite a saved-message; the repaired
capture verified a writable persistent directory and an actual file.

The DRAM timeline SHA256 is
`6331e3272df0325d2e9bea07012f158c35652e1976f0fcb7d1497a098d4a21a3`;
the SSD timeline SHA256 is
`f1c112a1b8049f69033aae18c81f6515e16884759eefc3cda5a7ebfc76ff1e0b`.
The DRAM trace contains 8394 kernels and 103 GPU memcpy events and includes
three profiled forwards. Their GPU activity spans 1392.68ms with a 59.20ms
GPU-event interval union.
Profiler overhead and one-rank sampling prevent interpreting the uncovered
interval as normal-service GPU idle time or a measured optimization opportunity.

P native ranks exited zero in the completed controlled runs. Some D native
ranks exited SIGABRT despite model roots/Consumers returning zero. The observed
DeepGEMM grid-sync/CUDA/SIGABRT chain was independently reproduced with an
unaltered official no-feature, no-remote, no-Consumer baseline. It is retained
as a known baseline limitation, not a claimed shutdown fix or proof of identical
underlying cause. Finite D runs also exited zero; they do not erase earlier
failures. Native exit codes and I/O drain are recorded separately.

Unknown DMA completion conservatively retains the real pages, operation and
registration owners until transport quiescence is established. A timeout is
not proof that buffers are safe to free, nor proof that a remote lease renewed.
This fault path may remain blocked; ordinary normal-exit evidence must not be
substituted for this missing quiescence guarantee.

## Bounded SSD writer result

One finite common-writer comparison reused the same saved 40448 real input IDs
and max_new_tokens=1 on the same P hardware, SDK and Provider. Both requests
had local0/remote0 and the same one-token output. Each rank wrote 395 objects
and 39,032,320 payload bytes; total payload was 156,129,280B in each version.
Warmup writes are separate. Profiling and copy/CUPTI observers were disabled
for both timed writers; the same exit-recording hook remained enabled.

| Version | Four native-rank KV_IO write durations ms | Median ms |
|---|---|---|
| 5daf singleton writes | 1570.052, 1570.059, 1570.031, 1576.294 | 1570.056 |
| ff13 bounded batches | 308.057, 315.226, 324.750, 377.426 | 319.988 |

The observed median write-operation duration decreased 79.62%. These are four
concurrent ranks in one comparison, not four independent experiments or a
statistical stability claim. The input int32-LE SHA256 was
`d881c0a67097dc4a73c2870513b524589b70f745751aad397dea97f2bb404f71`;
the identical output SHA256 was
`a72ea1862f9d3fac4a8bb0473b92b3891225cbb2bc6abd432abccf94d3e9a1d2`.
All eight necessary P native ranks exited zero across the two writers.

Server FinishWrite completion was observed for every transaction. Cross-host
request-start-to-last-server-FinishWrite observations were 1.85642s and
0.59261s; clocks were not calibrated sufficiently to treat these as exact
client-ACK latency. Same-PID KV_IO durations above provide the measured write
comparison. The candidate's D was already ready and idle before the writer;
its source request started only after writer completion. The baseline had no
D model. This is a common SSD writer improvement, not a real-D TTFT gain.

After the candidate writer, its distinct-prefix 1536-input request produced
513 real tokens through PD. D committed2048 with one new location and
1,976,320B. The test controller then shadowed its process-identity function
with a dictionary while starting fresh P; it failed before the new model
loaded. The original failure and newly spawned process's SIGTERM are retained.
The independent fresh-P continuation reused the same saved IDs and committed
checkpoint without rerunning the writer or source. It restored local0/remote2048
for2177 input tokens and produced32 tokens, but strict comparison against a
true zero-reuse bypass failed at index2: cached442 versus bypass1664. Both
inputs had SHA256
`27a7481f14965d15a8337319014299fa4048803fa752be9d3c9504c4712c3880`.
Cached output SHA256 was
`6ab1a3b3138d378660267797329f53e981b4a13ccfcc6e8f33419c793e18cd9a`;
bypass SHA256 was
`bfebc2f15fe11d56185095730f1cfd29d8a8441c9fa9ff8f701667ccb635cef1`.
This new D-source input differs from the earlier original P-only control. That
control does not establish an identical cause here, and no output tolerance was
invented after observing the result. Candidate P-side copy observation was
disabled for clean writer timing; do not claim a new P zero-staging observation.

The resumed P's four native ranks, root and Consumer exited0. D had one native
exit0 and one SIGABRT/status134 with the DeepGEMM grid-sync/barrier chain; its
root/Consumer returned0. The raw D harness incorrectly reported SUCCEEDED and
is overridden by the recorded native failure. Both GPU process lists were empty
after cleanup. Auxiliary SIGTERM/exit1 processes are listed separately rather
than being called native successes.

In that ff13 D publication, both native ranks registered seven GPU spans with
return0. Four outer SDK writes each carried494,080B, with durations
38.399/28.245/19.904/14.679ms and error0. D-local EOS-to-commit was105.368ms.
The covered registered KV ranges had D2D gather1,976,320B and no observed
H2D/D2H during publication. This is a separate short-input observation, not
a paired comparison against the earlier long-input D runs; existing observer
Graph/GetProcAddress limitations still apply.
The same Consumer's8192B positive controls passed in both directions, with
flush0 and zero dropped records; there were no business copy records outside
that positive control. This supports only the stated observation coverage.

## Review and remaining candidate checks

Actual dsh `bailian/kimi-k3` approved the 5daf source, subsequent test-only change
and the a16022d compatibility change. For ff13, the first review timed out after
1000s with no verdict; a bounded same-session follow-up verified the exact v2
delta and returned **APPROVE_CODE_ONLY**. An independent agent found no new code
blocker and checked the two source hashes and executed XMLs. Code review does
not substitute for runtime correctness. Fresh-P range restoration and cleanup
are complete, while the strict output gate is failed as described above.

Remaining: classify the new strict output failure using one minimal same-input,
same-hardware original-source P-only local-cache counterfactual, if the existing
warm build supports it. It will use unmodified official9115 source with the
official no-remote-cache build configuration, no SDK/Consumer/D, then seed2048
real saved IDs and compare cached/bypass2177 inputs. It is not a new D source or
performance run. Do not repeat the writer pair or long32K cohort to seek a
favorable result, and do not relabel the current strict failure as passed.
