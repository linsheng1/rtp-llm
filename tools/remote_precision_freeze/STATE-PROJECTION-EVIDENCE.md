# STATE projection precision evidence

Model candidate before this change: `61887b83eb180619032a46f0f2dc0eb86248ee09`.
Capture23 plan: `e9d7fc2c30a7c91080ae95ea00d52deed8bddf8141b85cb36c541fb1c98a6833`.
Input IDs SHA256: `07cf16434cfb7964bb1090f9c4c4d4b30823be0ef3bb3ace8c58fd4de913522c`.

A read-only comparison of the completed four-rank capture found all 17 L0/L1
query boundaries byte-equal, including FFN output and mHC coefficients. The five
sampled MoE query inputs were also equal; this does not establish whole-batch or
expert-dispatch equality. The earliest sampled difference was L2 STATE7167:
1022/1024 FP32 elements, maximum absolute difference 3.814697265625e-6. The L2
query7168 hidden input remained byte-equal. Normal STATE stores exactly matched
the original projection values/scores, and restored owner-pool bytes exactly
matched all four post-gather reads.

With the same captured input7167 row and original layer2 checkpoint weights,
five native cuBLAS calls exactly reproduced both complete STATE vectors.
Changing only M=7169 to M=1 caused the observed 1022-element difference;
combining the projection columns at M=1 made no further difference. Other rows
were synthetic zeros. This proves the observed GEMM shape phenomenon, rather
than equality of unobserved source-D inputs or a source/remote byte-transfer join.
Native counterfactual report SHA256:
`51aa8d85f5ab0ef8ace3e7591894bf86cb64764e7952eb8dd88bbeae4814b704`.

Five calls to the existing ad62 DeepGEMM BF16 GEMM, with deterministic algorithms
enabled and FP32 output, then produced byte-identical results across the same M
and N variants. All results exactly matched the original normal STATE and were
close to a CPU FP64 reference (rtol/atol 1e-5). Device memory stayed within the
original 1 GiB guard (about 850 MB used, including initialization). Accepted
component report SHA256:
`e3ad960aaac7d4d02271f300a7bf9f3ac82478b2e2e860499bd3462297544201`.

The change routes aligned SM100 compressor projections through that existing
backend. V4.1 initialization already enables its deterministic algorithms.
Other supported native paths retain the existing cuBLAS implementation. No
native library, SDK or complete wheel was rebuilt for this diagnostic.

This is component evidence only. The new module-level GPU regressions and real
normal/P-origin/D-origin complete-ID comparison must run against the bound new
candidate before whole-model repair can be accepted. Historical failed control
attempts remain in the task state; they are not counted as passing comparisons.

Actual candidate `91b4635faf8c65cc67e66ab52a46f5c932cd08ea` model follow-up (2026-11-02):

- The committed module's two GPU tests passed on C3, with zero skips/errors/failures. Both private Python overlays reuse the unchanged C1 native libraries and ad62 DeepGEMM; no wheel has been built for this candidate.
- Three model requests completed: original G7041 source returned241 tokens; D-remote returned113; the same-instance normal bypass returned143. All completed EOF below1024. Runtime EOS binding remains independently unverified.
- Both continuations use the exact same R7169 IDs as capture23 (`07cf1643…13522c`) and the same sampling parameters. P and D process epochs agree across these two continuation requests. D-remote reports local0/remote7168; normal reports local0/remote0.
- **Whole-model outputs still differ at output index18 (token19, one-based). This candidate is NOT a complete precision repair.** P-origin for91b has not been run. Full-wheel acceptance is blocked by this observed difference.
- The next finite batch is capture24, reusing the unchanged capture23 sampling points, observers and budgets for two requests plus all four rank drain/coverage proofs. Its frozen plan SHA is `fc5eff77515d4026b1dfc40314d1d866f8133e33d4ccf769dd5399d7bc9dc8e9`. Its actual module helpers are versioned against91b, the241-token source receipt and the143-token normal reference. The healthy D/Consumers remain; only P changes at a verified completed-request/drain boundary to establish a fresh instrumented allocator instance.

Safe device evidence references:

- Source: `control/state91b-source-source1/request/native/result.json`, SHA256 `57769b6b19d18558fbc3c472838b418be032642adf40a679fe61aa4b0fc643fd`.
- D-remote: `control/state91b-d-restore-native-fix91bD1/native/result.json`, SHA256 `d763a1cc363894deb801810dba73b9a9cddcdf75c2aae3deee502daa3213d46f`.
- Normal: `control/state91b-baseline-native-fix91bN1/native/result.json`, SHA256 `7347e87250dd2ab0e93c1bcc3955d190ff11eb333ce39565b4870505915e60a3`.

The static weight lease gap was resolved by a one-time device-local full checkpoint-vs-table hash verification of202,758,032,400 bytes per node, followed by new live read-only SH leases. No expired READY was reused, no table was unlinked, and no model/cache bytes were relayed through the Mac. The new model admission still verifies actual PID/UID/start, FD identities, READ FLOCKs, current heartbeat and full remaining lifetime. Posthoc native/Manager metadata does not prove exact source-D/remote/P byte provenance; that scope remains explicit.

capture24 completed both actual requests and both four-rank barriers. Its full IDs exactly reproduce the uninstrumented113/143 references, at the same token19 divergence. Empty proof age at request admission was17.586s, within the unchanged75s guard. Read-only comparison (`55e72120…810edbad`) proves L2 hidden7168 and all four restored STATE7167 copies byte-equal to normal; all17 L0/L1 boundaries and five MoE query boundaries remain byte-equal. L8 hidden7168 still differs5020/5120 (maxabs0.04296875), and its STATE7167 differs1024/1024. Thus the remaining calculation difference lies after L2 input/STATE and before L8 input; the specific operator remains unproven.

The next frozen batch, capture25 (`072f8070e194690d1cdce3f3eb4dae1042b4c79a62c138db0902d8a2c3c25c19`), adds only11 single-query boundaries covering that interval. Its unchanged2-request protocol uses at most2,834,848 additional retained bytes /2,906,592 peak bytes, below the original4MiB/16MiB limits. CPU model configuration passed; no capture25 request completion is claimed here.

All control plans use task-qualified basenames in the flat device control directory. The first capture24 analysis attempt correctly rejected an existing historical `analysis-plan.json` (`673a2228…91a2da9`), before any analysis launch or RPC. The old file was preserved; `capture24-analysis-plan.json` (`ab6892c1…6e86260`) completed through the same original controller. No model was restarted for this control error.

capture30 completed through the persistent frozen entry (2026-11-03 +08).
Plan SHA256 `61fdd8316ed1008de78fcd6d2e05fa6f7cfa277854dcf2802071a01e38969494`;
device actual report `control/runtime-freeze-capture30-actual.json`, SHA256
`f99765af8906e407575d54872c788d84fed1545b75901a3a6c5579a725386001`.
Actual CPU prepare/live identity gates passed on all three ends, two native
requests completed EOF below1024, and both four-rank barriers passed. Fresh
empty proof age was23.005856s within75. D-remote local0/remote7168 returned113;
normal local0/remote0 returned143. Both full ID lists exactly reproduce their
prior same-candidate references despite the documented P/D epoch change.
They still first differ at output index18/token19. P-origin remains unexecuted.
Candidate91b requests completed8; including618 task total16.

Paired epochs: P PID812208/start44723991, D PID760658/start44750076, UID500.
The old P and D reader models completed their original natural drain, actual
root returncodes0. Consumers, immutable Python overlays, native/ad62 artifacts
and static checkpoint tables were reused. No model restarted for metadata
errors. Earlier capture27/28 each had actual RPC0; their terminal evidence is
preserved separately. This is inner diagnostic evidence, not wheel acceptance.

All11 added query points were compared from existing raw files on the device.
Read-only report SHA256
`765dc5d4d61e038dc2fe3c4794a359574fea41b66529c3db62beae60ab71297a`.
The earliest newly sampled difference is **L2 attention projected+all-reduced
output7168**,4458/5120 BF16 elements, maxabs0.0205078125. L3 normalized input
already differs4588/5120. L2 hidden7168 and prior STATE7167 remain byte-equal;
all17 L0/L1 query points and five sampled MoE query inputs remain equal.
These are sampled boundaries, not whole-batch or complete-history equality.

A second read-only pass reused the same four-rank capture and compared original
L2 values512/scores512 at7168 (all16 cross-rank pairs each), plus STATE store1024
at7168 (actual valid owners). Every comparison is byte-equal. Both cases took
prefill=true, batched_groups=false, projected_full=false, previous-state-present.
Accepted report SHA256
`33e9f577de1050764e3db54e7c717114afc245ed6e221853d28caf8875728fdb`.
The first postprocessor failed before publishing a result because a bytes loop
variable shadowed argparse arguments; only that variable was renamed in a new
immutable source/plan. This added zero model requests and no GPU rerun.

Specific remaining gap: actual Q/selected historical KV, sparse index and
length mapping, sink/RoPE/static parameter identity, core output and projection
before/after all-reduce inside L2 attention. No evidence yet attributes the
remaining difference to transfer, selection, a kernel or output projection.
No further arithmetic change or wheel build is justified yet.
