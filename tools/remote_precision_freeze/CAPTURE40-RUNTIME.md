L25 Q/core/readset diagnostic continuation
========================================

Candidate031fc2af33111e7bbe13bbca6e72ebd6fc094d7c remains the production source. No model math patch or wheel build. All devices C3 only, P12/D71, existing separate storage73. All data stay device-direct; only safe metadata/control files leave devices. Current shared state is deployment/state/publication-precision-20261101.json, HC_candidate031_20261103.next_capture40.

Evidence entering this batch
----------------------------

capture39 two real requests naturally completedD167/normal176, exact original IDs/profile. First P7168 and D7169 unequal Block entryL26. L25 D actual attention input, prepared HC coefficients and shared quant buffers are equal, but attention output differs4583/5120. Its FFN seams differ later. Actual L25 topology: ratio1/KVsource20/indexsource24/TP1/64heads. No FFN patch justified.

Profile40 captures only natural D L25/T7169: actual input token, Qr/Q and quant buffers, actual pre-coreQ, selected rawSWA528/GLOBAL288, physical and inverse-block-table logical indices, validity, request tables/start/length, actual sink/cosSin row, core output/scales, projection and pre/postAR. Original4MiB/192object/256KiB-point limits retained. P has no hooks. All6 actual startup target modules passed CUDA-hidden import/source checks. Two new graph readset contracts passed, zero skips/errors/failures, under original emptyGPU/1GiB limits.

The first unit attempt failed beforeTorch for nonzero65536 used_v2/no compute apps. Later original audit saw2621440. A subsequent full4GPU inventory showed all used_v2=0/no apps. Unit v2 then passed with the same thresholds. No model was restarted to solve this.

Finite dependencies
--------------------

Old model epochs39 naturally ended root0/allranks gone. Consumer3 idle handoff -> same Consumer4 binary/SDK/ports, new consumer91b-P/D-v4, actualREADY/REQUEST_BLOCKEDfalse, four-hour leases. Current nativeP971332 UID500/start47812306, D905010 UID500/start47813752; these are snapshots, not permanently live claims. New keeper hc031-static-P/D-v2 fully hashed202758032400 source/table bytes each, read-onlySHFDs held; old v1 retained. Historical405516064800 total read bytes/node took238s, and new full validation completed under original controller. No cross-cluster or Mac payload transfer.

Persistent entries
------------------

- `prepare_capture40_controls.py`, `prepare_capture40_models.py`: new profile, actual CPU compose/check, immutable role plans, original submit/controller/drain wrapper.
- `preflight_capture40_hooks.py`: actual target startup import with CUDA hidden, no model/RPC.
- `renew_consumer_boundary_v4_entry.py`, `renew_hc031_static_v2_entry.py`: original safe idle Consumer handoff and full-content keeper verifier; no binary/port change.
- `freeze_capture40.py`, `launch_capture40.py`, `collect_capture40.py`: actual recipe/freeze/original check-render3argv/preflight/submit; one bounded status read.
- `recover_capture40_entry.py`: zero-native-RPC recovery of already-published typed latches; the original parameter-hash failure remainsFAILED.
- `freeze_capture40_typed_v2.py` (N2), `freeze_capture40_typed_v3.py` (N3), `launch_capture40_typed_v2.py`: explicitly typed-core-only normal request; all original identity/45s heartbeat/180s timeout/75s proof/once/native EOF gates retained. No accepted GPU projection fingerprint claim.
- `prepare_capture40_successor.py`: unchanged sampler/runtime/SDK/inputs in successor finite model epochs after old epochs expire; no D request replay.
- `capture40_compare_typed_entry_v2.py`: CPU-only actual core-data comparison. Cross-epoch causal conclusions require exact singleGPU reproduction of both actual core outputs before a counterfactual.

Actual D1 and export failure
----------------------------

D1 plan SHAa65cf05817e6d37795af137cb49a5b4636403110001823d73aea070e7d693c77. Native request completed167/EOF/exact original IDs/profile, resultSHA0eb16208b01f24577121e738b4ca26205e826e5e738c65520c101fcc80fcf49e, RID169189659820113. Task total36/candidate031 total14 real completed requests at this checkpoint.

Original stage FAILED afterEOF: GPU parameter fingerprint total exceeded original128MiB cap. Original counter loop checks every count0/1, then publishes typed.pt, then computes parameter hashes. The published file is valid core-data evidence; GPU projection parameter fingerprints are stillBLOCKED. No cap was raised and no live dictionary/model hook was modified.

Zero-RPC recovery planSHA6b3c98ecef2b08cb63bf08d28772d6ef12ca6641b5f7710e73c66c657ab17057 verified original immutable latch source and publication ordering, actual rankUID/PID/start/config, nativeEOF/token/position, fileUID/type/size/SHA and original bounds. It recovered four rank files (one actualDP owner908105 with37 points, other3 empty), resultSHA73a226c9f6e78e08fd63275e83857fa6678da02b0628211d65792f9f400f177d. State is SUCCEEDED_TYPED_RECOVERY_ONLY, not an original-stage or parameter-hash pass.

Normal attempts and exact commands
----------------------------------

OriginalN1 never submitted; its original dependency remainsBLOCKED. N2 planSHA06a81a66c9ab0b206e9e23f12e102e9dda0df9a7f4f939698c5cb781c8018783 used an explicit typed-only export adapter. It was refused by the unchanged READY/lease identity check before request-once; actualRPC0. No timeout/age/lease threshold changed. Old model40 epochs expired naturally.

N3 reuses the completed D1 data and sends only one normal request in unchanged-runtime successor epochs. Its different D epoch must be reported, never hidden. Candidate, input, generation profile and control/source identities remain checked. Model epoch plans: Pf460cea90ccd995224dab92bc53e6914e80f935dd36702b37615b873d9f744d5, Da43d1b4d934d1808c7e050e7703fc5df69791dbb0ad08afdab3ca4100b999d53. Already submitted once; do not replay:

```sh
E='/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze'
D='/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/capture40'
python3 -B "$E/launch_capture40_typed_v2.py" --plan "$D/N3-plan-v1.json" --sha256 433ab699ee873c97b919e2543796bd1007902dc1cd41534fa973c7970514f19c --evidence-dir "$D/N3-launch-v1" --submit
```

N3 normal completion requires176/EOF/exact reference IDs/profile and four rank typed exports. A typed-only result does not verify GPU projection hashes, full historyKV, original businessR, immutable wheel or whole-model repair. Freeze a new uniquely batch-qualified CPU plan filename; do not use the old global CPU-compare-plan-v1.json basename.

Completed continuation (2026-11-03)
----------------------------------

N3 normal naturally completed176/EOF/referenceIDs exact and all4ranktyped exports; nativeSHA cc178ece76b99fadd68bca28726f090daa3d2241e61f3ad551a5f2ffdeff6bf9. Task total37, candidate031 total15. N2 remainedRPC0; no native replay. Original128MiB parameter-fingerprint failure remains explicitlyBLOCKED.

Typed CPU comparison resultSHA e84dec21c41583f2e17c0e0881d0c1a9332254fc1c8205f891f8db705f86ba21: Q/Qr/inputquant/sink/cos/queryposition/readsets equal; GLOBAL512 rawrows equal. SWA90/128 rawrows differ4719bytes; actualcore359bytes and projection4583/5120 differ. CrossD epoch is recorded.

`capture40_swa_replay_entry.py` uses recipe->freeze->check/render->originalsubmit/controller, samePID physicalGPU0-empty gate and original1GiB limit. Actual CPUcheck PASS; runplan247502052fd59fd32135882241e9c599e2a3bf1b2668bc6a6f04d0178ac6f944 completed2 primitivecalls/0 modelRPC. Replacing only capturedSWA bytes exactly reproduces both actual coreoutputs AND scales, resultSHA483a08cf680ab9128ea177041ce61680f92fb2f2fb3f1bfd68d0171110ce8557. No kernel/model patch, no wheel, no projection proof claimed.

CPUfield comparison (`capture40_swa_format_entry.py`) plan5b5c7524cdc61e52a69a9b73aae8170891c670b656155bc67b7620218c84473c passed. Result7d048c814a10a5fc41c95b2f57d893ed16da654904deec1b63ee05596c359443: 138 groups change32payloadbytes+1exponent but decode exactlyequal;89 rawchangedrows have equaldecodedvalues. Only1 row differs numerically:165/512 elements,max0.125. Absolute token mapping not yet proven. SWA pool is percurrentlayer; GLOBALsource20 must not be interpreted asSWAsource20.

Persistent rowcounterfactual entry `capture40_swa_row_replay_entry.py` isolates the1 numericrow from the89 decoded-equivalent encodings. It requires prior exactcalibration, independentactualCPUcheck, newimmutableplan and originalempty/1GiB admission. No existing experiment is rerun. Current models ended naturally; Consumer4/staticv2 remain underoriginalfinitecontrollers. Remaining gap: prove origin ofthe actual numericSWAdifference, then realthree-route IDs; normal176=P-remote176,D-remote167 diverges atzero-based32. DiagnosticR7169 is not originalbusinessR or immutablewheelacceptance.

Row isolation and position join
-------------------------------

Rowcounterfactual plan965401c5dab0f5bb42fbd51fe2614c3ca99f456916f2b89fd3d04db2624c0876 completed2 calls/0RPC, result3a09ee31c1b084035e9f3e2a9da9b8c7741d05df9d0495207d1658ceebe173a0. Source numericrow + normal equivalentencodings reproduces actualD output/scales exactly; normal numericrow + source equivalentencodings reproduces actualnormal output/scales exactly. The numericrow alone is sufficient. Rowordinal3/inversecoordinate6661; no representation-only cause claimed.

Positionjoin plan652c75138b50fc863ba99e7217e0c1ddff142bc062807f44daac65ebdaaf7ccf completedCPU/0RPC, result83503b1360e2ff74fb6fed4e4a324a1723be729f835b0b7567a9296f03c471bf. For both actual captured windows, all128 physical slots and inversecoordinates match production512-token block-table span and128-slot ring/window formula. Ordinal3 maps to7045. This is a slot/window mapping proof; historical producer-token/value identity and source/remote/destination byte equality are still unproven.

Accurate existing commands (check/render only; runs already submitted once):
```sh
python3 -B "$E/capture40_swa_replay_entry.py" check --plan "$D/Capture40-SWAReplay-run-plan-v1.json" --sha256 247502052fd59fd32135882241e9c599e2a3bf1b2668bc6a6f04d0178ac6f944
python3 -B "$E/capture40_swa_row_replay_entry.py" render --plan "$D/Capture40-SWARow-run-plan-v1.json" --sha256 965401c5dab0f5bb42fbd51fe2614c3ca99f456916f2b89fd3d04db2624c0876
python3 -B "$E/capture40_swa_position_entry.py" check --plan "$D/Capture40-SWAPosition-plan-v1.json" --sha256 652c75138b50fc863ba99e7217e0c1ddff142bc062807f44daac65ebdaaf7ccf
```

Next sampling, if required, targets actual historical position7045 in the originalG7041 source decode and normalR7169 prefill, finding the earliest different entry/actual normalized attention input throughL25. Sampling only7168/7169 cannot establish those historical inputs. No oldattention primitive experiment or completed native comparison needs replay.
