Capture44 continues candidate 031fc2af33111e7bbe13bbca6e72ebd6fc094d7c on C3 P12/D71/storage73 only. Root Codex is the sole executor. No DSH, additional agent, wheel build, install or new scheduler.

The prior capture43 source failed before native RPC: the observer context omitted `layer`. D's original FAILED/rc1 cleanup and P's original finite-lease CANCELLED/root-rc0 drain remain preserved. Capture43 contributed zero requests. The task had 41 completed native requests before capture44.

Persistent entry (workspace absolute):
`/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze/prepare_capture44_models.py`

Frozen inputs live under `/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/capture44/`. Never overwrite an existing plan/config/receipt, substitute example v2 values or edit old SHA fields. Each command below uses the recorded SHA from the exact frozen plan.

CPU/full-chain prerequisite: `capture44_chain_entry.py check|render|submit --plan …/Capture44-CPUChain-plan-v1.json --sha256 29a1ad263d0c4615317449581c14307d34dde1f3003e2fd8ba98a4ba3a079dae` (already SUCCEEDED; do not resubmit). The synthetic CPU boundary virtualizes CUDA presence and synchronization, while actual wrapper/context/record/latches/socket/export/typed comparator remain real.

GPU prerequisite: `capture44_unit_entry_v2.py check|render|submit --plan …/Capture44-GPUChain-plan-v2.json --sha256 7be8d53fef9854913f88db9aea1667fb9a99c944d764a772c9564c488a4b51f6` (already SUCCEEDED; do not resubmit). The original empty-GPU0/same-PID/NVML/1-GiB admission remains. Actual P is eager; D uses graph capture/replay. The earlier extra-scope P-graph failure is retained, not hidden or described as a model failure.

Model preparation (already completed; do not repeat immutable publication):
`python3 tools/remote_precision_freeze/prepare_capture44_models.py prepare --recipe …/capture44/models-recipe-v1.json --directory …/capture44/models-v1`

Model startup uses `prepare_capture44_models.py check|render|submit --plan …/models-v1/{P,D}-model-plan.json --sha256 <outer-plan-SHA> --receipt <new-receipt>`, then the original controller. P outer SHA: `42d7dc3c936ffc9816c775ac062050268c5f4f965c797842ada6e204fdfa21d6`; D outer SHA: `59d1776ac196951cc0a94347e1b5780c2f4bb641a29d42412fddd1b5f2cc2af8`.

Actual paired request stages:
`python3 tools/remote_precision_freeze/launch_capture44_source.py --plan …/Capture44-Source-plan-v1.json --sha256 6d75a05143e74a232c55d2fad4090084ba831deb4ee251fefeff65337318714e --evidence-dir <new-dir> --submit`
then only after source EOF/calibrated295 IDs + all4rank exports:
`python3 tools/remote_precision_freeze/launch_capture44.py --plan …/Capture44-Normal-plan-v1.json --sha256 9d290609acb7d388bd77c03d46a231386a8f02ce153d06367edf8fed8a260ef4 --evidence-dir <new-dir> --submit`.
Both routes remain freeze/check/render → original submit/device controller; device-side real prepare, paired native probes, READY/PID/start, leases and fresh-P proof retain their original validity windows. Dispatch/READY is never completion. Inspect device terminal state before replaying any ambiguous SSH submit.

The two requests are the unchanged G7041 source295 and normal R7169 continuation176, natural EOF/full reference IDs; no expanded matrix. The complete stage ends with both EOFs, four-rank evidence and `capture44_compare_entry.py` CPU comparison. L24 Qr/quant/Q, actual selected canonical KV/per-pool order, core/scales, output projection/AR are compared. Existing capture42 L24 normalized input equal / attention output different evidence remains valid. No independent FFN/HC cause or whole-model repair is claimed.

Whole-chain active allowance: load/READY240 + prepare120 + paired probes120 + two RPCs360 + exports300 =1140s, within original1380s active service; drain420s is reserved separately. Original handoff240 + service1800 +15s dependency margin is audited before startup. Consumer5 and full-content keeper3 are retained; the model admission repeats original live checks.

Overall termination remains three-route complete token IDs equality, necessary regression, and commit/artifact-bound report. A control-chain pass is not model numerical acceptance.

Executable commands from any shell (absolute paths; inspect receipts before submit):
```sh
C44_ENTRY='/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze'
C44_BATCH='/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/capture44'
python3 "$C44_ENTRY/prepare_capture44_models.py" check --plan "$C44_BATCH/models-v1/P-model-plan.json" --sha256 42d7dc3c936ffc9816c775ac062050268c5f4f965c797842ada6e204fdfa21d6
python3 "$C44_ENTRY/prepare_capture44_models.py" check --plan "$C44_BATCH/models-v1/D-model-plan.json" --sha256 59d1776ac196951cc0a94347e1b5780c2f4bb641a29d42412fddd1b5f2cc2af8
python3 "$C44_ENTRY/launch_capture44_source.py" --plan "$C44_BATCH/Capture44-Source-plan-v1.json" --sha256 6d75a05143e74a232c55d2fad4090084ba831deb4ee251fefeff65337318714e --evidence-dir "$C44_BATCH/source-launch-v1" --submit
# Only after the source's terminal EOF/calibration/exports check:
python3 "$C44_ENTRY/launch_capture44.py" --plan "$C44_BATCH/Capture44-Normal-plan-v1.json" --sha256 9d290609acb7d388bd77c03d46a231386a8f02ce153d06367edf8fed8a260ef4 --evidence-dir "$C44_BATCH/normal-launch-v1" --submit
```
These directories/attempt names are immutable. Existing successful launch receipts mean inspect/join, never start them again or choose a new name to bypass an existing failed state.

A repeated source-publication check encountered SSH255 before any stage launch or RPC (storage absence was verified). The healthy models and completed CPU prepare are retained. Use this persistent submission adapter to verify all existing device references in one read per role, then render the exact frozen argv into original submit:
```sh
python3 "$C44_ENTRY/submit_capture44_prepared.py" --plan "$C44_BATCH/Capture44-Source-plan-v1.json" --sha256 6d75a05143e74a232c55d2fad4090084ba831deb4ee251fefeff65337318714e --prepared-directory "$C44_BATCH/source-preflight-v1" --evidence-directory "$C44_BATCH/source-launch-prepared-v1"
python3 "$C44_ENTRY/submit_capture44_prepared.py" --plan "$C44_BATCH/Capture44-Normal-plan-v1.json" --sha256 9d290609acb7d388bd77c03d46a231386a8f02ce153d06367edf8fed8a260ef4 --prepared-directory "$C44_BATCH/normal-preflight-v1" --evidence-directory "$C44_BATCH/normal-launch-prepared-v1"
```
The adapter pins its own source, stage plan, prepared proof and rendered argv in an immutable submission-plan.json. The original device receivers/senders perform full live prepare and native admission. Existing attempt receipts must be joined, not repeated.

Completed evidence: two model requests (source295/normal176), both natural EOF/full reference IDs and four ranks per role; task total43. Comparison v1 failed only because D's actual merged-QB path bypasses Mega.project_q. V2 uses the actual returned D Qr (`Qr_from_qkv`) for the same normalized-Qr comparison and preserves absent D quantization hooks as unavailable. No model rerun occurred.

Comparison plan `cab5ec036caba3de40a6a4a61848d1bc1448b44210e8b0b162f043e9d301e6ae`, result `f104e780ab890707e1698165c9776df0ee5448ac4fcc2794e45ab158dd1ec714`: actual input/Qr/Q/positions/sink/cos/scale plus all11parameter fingerprints equal. SWA128 rows numerically exact. Core FP8 encoding differs267bytes/32768 with exact scales; projected BF16 output differs4500/5120,max0.0390625, exactly reproducing capture42.

GLOBAL audit plan `796a06bd7373b011a52e542c1590ea4e0bb6392c156a913f74dab2dbec2a8d50`, result `c3e57d715d0cb95ad1487c056757a19fafe03b245f8863ce105697b37433e092`: 511 common canonical GLOBAL rows have exact BF16 values. Each route selects one exclusive row; those decoded values differ, and selection order differs. Actual mapping checks pass; parameter hashes cover90,025,728bytes per target rank, below128MiB. This identifies an earlier input/readset difference, not a proven kernel or selector root cause.

Next narrow primitive stage: `capture44_global_replay_entry.py` freezes/checks/renders/submits at most3 calls after CPU input preparation. Calibrate normal actual P output/scales and source actual D output/scales using their captured decoded readsets through the same P primitive; block the intervention if either fails. Only after both reproduce, replace the single normal-only GLOBAL row with the actual source-only row, keeping normal order and all other inputs. No further model request or matrix expansion. D model may be drained only through `drain_capture44_entry.py` after the verified two-EOF/four-rank boundary to free GPU0 under original empty/1GiB admission; retain P's original lease, Consumers and keepers.

GLOBAL single-variable stage completed with exactly3 primitive calls and0 new native requests. Run plan `4457b01a1273611559ecd38984021f40382f594de27bf44f76bbe43c8c5838d8`; result `ce803280578e66b57b35cac30fb51bdd2b8c3f51b98477ad085c1fe2ded4404e`. Both actual P/D FP8 outputs and scales reproduce exactly. Keeping the complete normal order and all other inputs, replacing exactly the exclusive GLOBAL row at selected-row ordinal226 reproduces the actual D output/scales completely. This proves that one membership row is sufficient for the observed core difference; it does not prove why the indexer selected a different member or whole-model repair. No further L24 attention primitive matrix is needed. Next scope: actual L24 indexer inputs/keys/scores/selection at7045.

The original240s drain waiter FAILED before D's original model controller later completed rootrc0/cleanupFINISHED. GPUv2 was recorded BLOCKED with0calls. A new actual final-PID/emptyGPU0 audit resolved the externally changed state for planv3; old failure preserved, no wait/lease/1GiB threshold changed, no environment/model restarted. CPUreplayv1's wrong GLOBAL-first assumption was caught before GPU work; v2 uses actual SWA-first ordering.
