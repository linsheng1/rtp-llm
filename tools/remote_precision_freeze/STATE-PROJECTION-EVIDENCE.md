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

capture34 reused the candidate91b environment and completed two real requests,
both EOF below1024, plus both four-rank typed attention barriers. Frozen plan
SHA256 `cc7c065360644b17fba49353f36519229bb8fe858ff545609313e92f31dd0107`.
Actual report SHA256
`8746d887507ac2f156942aa4fbb7e70e01295bbcdf1c08b36d3f3c575994c8c1`.
The complete113/143 IDs again exactly reproduce their prior references and
first differ at output index18/token19. Candidate91b completed requests10;
including618, task total18. P-origin91b remains unexecuted.

The read-only comparison of these existing raw captures completed with report
SHA256 `2628a46cd9ba1d72e8e407a1d4640eae882570b9d789dc779d9036c79c67d0eb`.
Both cases use mega attention. Qr1280, projected Q, all actual static parameter
fingerprints and the ordered640 logical KV addresses are equal. However, six
global rows and eleven SWA rows differ in347 BF16 elements. Their logical token
positions are within7041..7167, the expected D-new interval. This is logical
attribution only; equality of historical model inputs and a source/remote/P byte
join have not been established. The first differing selected row in logical
order is global3521, covering the pair ending7043 (two unequal elements).

The original core FP8 bits differ in526 elements, packed int32 scales in one
element, and projected output in4458/5120 elements (maxabs0.0205078125).
TP size is1 at this boundary. The core therefore receives different KV values;
this evidence alone does not identify a defective core kernel, producer,
codec, or transfer. Snapshots are at sparse-attention entry, not an assertion
about all historical inputs, unused workspace rows or later aliasing.

P837851/start45254691 and D781166/start45254956 (UID500) are the paired request
epochs. P34's outer controller subsequently ended TIMED_OUT/rc-9: its1500s
wrapper was shorter than the service's1800s total lifetime. All requests and
capture barriers had already completed; the model and four ranks are now gone.
The missing service cleanup result must not be described as natural drain or
successful shutdown. Preserve this failed terminal evidence and correct the
wrapper lifetime accounting before any next model epoch. Do not restart models
to repair this metadata. D reader6 and Consumer v2 are separate finite services.

The next bounded probe uses `capture34_kv_replay_entry.py`: freeze a recipe,
check/render, then the original submit/controller. Its CPU stage verifies the
existing four-rank proof, Q/KV/raw references, actual sink checkpoint bytes and
the full reconstructed RoPE table against captured parameter SHA256. Its GPU
stage retains the existing strict empty-device admission and1GiB guard. At
most three core calls calibrate both original query outputs and then replace
only the17 unequal selected KV rows. Failed calibration blocks attribution.
No model request, new environment, wheel, old STATE/mHC experiment or producer
provenance claim is part of this probe.

The finite probe has now completed: CPU plan SHA256
`930f8c232755f73836537c697883d45d3edc11d81152f5b1ee2bd56c388e8312`,
GPU plan SHA256
`19d63b9d728c199b26c7d55081620dc5689eae3dd61b11b17f362e83c12f346f`.
All three actual core calls finished with controller rc0. The first two exactly
reproduce both original FP8 outputs and packed scales. The third changes only
the17 unequal selected rows in the short workspace and exactly produces the
full-reference output and scales. Physical FB use849,674,240 bytes stays within
the original1GiB guard. No model RPC was issued; total model requests remain18.
This proves those KV differences are sufficient for the observed core delta.
It does not prove why their historical producer values differ. Original query
rows are reused; other queries and unselected workspace rows are synthetic.

Persisted command (already executed; do not replay this immutable batch):

```bash
python3 -B deployment/branches/publication-precision-20261101/tools/remote_precision_freeze/capture34_kv_replay_entry.py submit \
  --plan deployment/coordination/runtime-freeze-batches-20261102/capture34/capture34-kv-replay-GPU-plan.json \
  --sha256 19d63b9d728c199b26c7d55081620dc5689eae3dd61b11b17f362e83c12f346f \
  --receipt deployment/reports/publication-precision-20261101/capture34-kv-replay-GPU-submit.json
```


## Historical producer capture35 (91b, inner diagnostic)

Two additional real requests completed below1024 with native EOF: source7041
returned241 IDs, normal7169 returned143. Both complete outputs exactly match
their earlier references. Candidate91b completed12, task total20. No new
Dremote/P-origin acceptance was run. Normal143 vs prior Dremote113 still first
differ at zero-based18; whole-model repair is incomplete.

Source plan `4045921263fd8c08087498c8ecbac8dce6d98b0cecd502d27bf2943adc017624`;
normal plan `cec5f9079a27dfa3e4c4d5678759d82ce235f936212f84365226360080903700`.
Native result hashes are respectively
`2141067423f6fa268786f7585717f0f5b4338c5a8206b7735d1d424368cc9a92` and
`03180d2d6712b3c9ee39397686a5627ad9874686a71003436f0e982cd9417d38`.
All four ranks exported and joined actual sampled GPU token positions7042/43
with the sole native request history. Empty DP/CP ranks remain explicit. Raw
tensors/IDs went device-direct to the seed; this repository stores safe counts,
hashes and coordinates only. Source and normal model epochs differ and are
recorded in `history35-comparison-safe.json`; both whole-output calibrations
are required before comparing their samples.

CPU analysis v2 plan
`150f0df5c62938a731cf9c1a97a392254a2109f57fab6c8a2445ad9704325770`
finished under the original device controller with rc0 and zero new model calls.
Report SHA256
`e10cf0d54719da1830591fc9b7c20846943bdb162002b5acb5343be410eeb663`.
L0 attention and FFN input/output, and L1 attention input/output are byte-equal
at both positions. Earliest ordered sampled difference is L1 FFN input7043:
1/5120 BF16 values, column1958, maxabs0.00048828125. FFN output then differs
1080/5120. L2 input differs511/5120; compressed values/scores each differ512/512;
SWA decode differs69/512 and GLOBAL raw288 differs2 bytes. Position7042 is equal.
This is upstream of L2 compression/transfer; neither is established as root.

The existing L1 block-entry4HC also matches20480/20480 values. However layer1
has Engram and retained fusion is0: entry is before injection. Its equality and
normalized attention-input equality do not prove the actual post-injection
residual, HC coefficients, routing/expert inputs, quantization scales or whole
batch equality. Next capture is limited to that missing natural Engram→HC→norm
seam; do not repeat prior L0 split/STATE experiments.

Control failures before requests are preserved: eager P latches are created on
first natural CP forward; strict lazy identity still binds all4 live ranks.
Inference tensors cannot be reset outside inference mode, so source uses strict
read-only all-zero counter proof. Baseline preparation consumes actual successful
CLI stdout rather than a nonexistent summary filename. New source/core versions
retain all immutable historical fingerprints. Healthy models were not rebuilt
for these metadata failures. Later model batches use the original after-drain
wrapper with complete wait+service+cleanup lifetime budget.

Persistent executed entries (do not replay an immutable completed batch):
`freeze_history35_source_v3.py` + `launch_history35_source_v3.py`;
`freeze_history35_baseline_v2.py` + `launch_history35_baseline_v2.py`;
`history35_compare_entry_v2.py freeze/check/render/submit`. The shared state
contains the actual plan/receipt paths. Offline checks do not replace device
prepare, current leases, process identities or native request-once admission.


## Missing L1 seam capture36 (91b, inner diagnostic)

Both requested new model calls completed native EOF below1024 with all four
rank exports. Source7041→241 and normal7169→143 exactly match their complete
prior ID references. Candidate91b completed14, task total22. Source and normal
use the same P881261/start46291568 and D823739/start46292451 (UID500) model
epochs. Normal uses explicit bypass with actual local0/remote0; this is not a
fresh-P proof for a remote restore. P-origin acceptance remains unexecuted;
normal143 vs prior Dremote113 still diverges at output index18/token19.

Source plan `26d9acca69458558a90d67d3ccb615b061a0199188d287251b4bd6b1de863515`,
native SHA `041ab911d8061900d937877742fee0d5577cdb60878cd21c951e25b164cc33d9`;
normal plan `c58e212b60df5c6e904cdfe7fb23fc6875de531280cd237bf7da530204296072`,
native SHA `69c07d12329575f89673904d28686145c7013f2120b82375e17522419fb1e744`.
CPU comparison plan
`0ddf9924001e054c8fe63f491a6a948cee50c0b32d0c6048d437d716f3fa716e`
finished rc0 with zero model calls; report
`a225a1cfbb42c6100f0b6e180291f974e6b0ce7937595aa74750b10e1e694a57`
is stored in `history36-comparison-safe.json`. Raw tensors/IDs stayed on C3.

L1 post-Engram residual and actual attention-HC residual match20480/20480 BF16
values. All seven actual owner parameter fingerprints match (attention/FFN HC
fn/base/scale and FFN norm weight). Actual row count is sampled as a selected
GPU integer: source1, normal1794. Registration metadata is not used as actual M.
Earliest ordered sampled difference is attention HC pre at7042:4/4 FP32,
maxabs5.960464477539063e-08. At7043 pre differs3/4 and comb13/16; post4 matches.
Attention5120 input/output still match. Both intra-layer seams actually return
the fused path. Stored post4HC and all new FFN-HC pre/post/comb match at7043,
while fused FFN normalization differs1/5120 BF16, column1958. Current attention
pre-mix is the delayed input to this FFN collapse; new FFN coefficients belong
to later delayed computations. Mutual causality of concurrent outputs is not
assumed. Operator attribution and whole-model correctness remain unproved.

No healthy model was restarted for control failures. Earlier failed unit proofs
are retained: existing fixture directory; subsequently overlong socket path;
then strict empty-GPU admission saw2621440 bytes with no compute PID. The next
independent check showed0 bytes and both old unit epochs absent. The original
zero-FB and1GiB guards were retained. Final unit plan
`57df4dabe9948c8c300f41445e5800414b68fdee75f58c0563f00e4ec06feb87`
passed all4 contracts,0skip/fail/error. P's SSH255 was checked on-device: no
static-v3 plan/spec/launch existed before the original immutable plan was
submitted once. Both static-v3 full-content checks passed; Consumers v2 stayed
healthy. No DSH/new scheduler, binary rebuild or wheel was used.

Persistent successful workflow: `prepare_history36_models.py`;
`freeze_history36_source_v1.py` → `launch_history36_source_v1.py`; actual source
receipt → `prepare_history36_baseline.py` → `freeze_history36_baseline_v1.py` →
`launch_history36_baseline_v1.py`; `history36_compare_entry.py`
freeze/check/render/submit. `read_history_stage.py` performs one safe bounded
read; device controllers retain all periodic monitoring. Shared state contains
actual recipes, plans and exact commands. Do not replay completed immutable
batches. Model services keep their original drain and Consumer/static lease
guards; component replay waits for actual owned model drain/empty-GPU admission.
