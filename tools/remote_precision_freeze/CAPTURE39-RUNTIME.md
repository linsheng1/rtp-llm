Prepared-input trajectory continuation
=====================================

Candidate031fc2af33111e7bbe13bbca6e72ebd6fc094d7c, C3 P12/D71 and separate seed73 storage. Diagnostic actual-D-history R7169, not original business R or immutable-wheel acceptance. No production math changed; no wheel built.

Why this profile exists
-----------------------

capture38 found first unequal D Block x at L26/T7169 and L27/T7200. `forward_decode_layer` computes cross-layer MegaMHC before the Block: the prepared residual can match while normalized input, HC coefficients or quantization differ. Residual equality alone does not isolate an operator inside L25.

This profile samples P query7168/all40 Block entries; D7169/Block entries, prepared presence, prepared normalized input/24FP32 HC coefficients and actual attention input through L26; focused L25 attention output, actual shared quant buffers, HC/FFN seam and FFN input/output. Original4MiB/192object budget, original CUDA Graph setting and private request-drained reset/export remain. The actual first restore exported41 valid P points and148 D points, one active owner each, other3rank outputs explicitly empty.

GPU preflight and safe boundary
-------------------------------

Both capture38 requests had exact reference IDs/profiles, EOF and all4rank exports before only the owned P stop marker was written. Original controller drained P root0/all4ranks absent. D retained until its original natural deadline. Consumers/static keepers/storage retained.

The first diagnostic-unit bootstrap rejected used_v2=2621440 with no compute apps before importing Torch. Failed handoff1190 unit/483390 handoff retained; no model RPC. Fresh NVML then saw728039424 used/no apps. An independent Compute/Graphics/MPS plus owned-GPU-fd audit subsequently found all4cards used_v2=0/all lists empty. New unit attempt output/claim v2 passed actual3 graph contracts with no skips/errors/failures and unchanged strict0-before-runtime/1GiB-after-runtime guards. No model rebuild or threshold relaxation was used to fix the admission failure.

Persistent entries
------------------

- `prepare_capture39_controls.py`: versioned helper binding generation; no device operation.
- `handoff_capture39_unit.py`: completed-stage SHA/epoch checks -> owned P marker -> original drain wrapper -> strict unit bootstrap, all supervised by original controller.
- `capture39_unit_entry_v2.py`: frozen isolated unit attempt. v1 handoff failed admission; v2 actual3 tests passed.
- `prepare_capture39_models.py`: real CPU compose/check, immutable epoch plans, original submit/drain wrapper.
- `freeze_capture39.py`, `launch_capture39.py`: recipe -> freeze -> original check/render three argv -> real CLI/source derivation preflight -> original submit/controller -> live identity/75s proof/full prepare/once/EOF/export.
- `collect_capture39.py`: single bounded read, no RPC or polling loop.
- `capture39_compare_entry.py`: complete receipts -> frozen CPU-only comparison via original controller. Original raw data stay device-direct.

Executed batch commands; do not replay
--------------------------------------

```sh
E='/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze'
D='/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/capture39'
python3 -B "$E/launch_capture39.py" --plan "$D/D1-plan-v1.json" --sha256 17522ad4d22ca346a89567ad4d72512a4b0b571029f4dedffa74bbb68eb28926 --evidence-dir "$D/D1-launch-v1" --submit
python3 -B "$E/launch_capture39.py" --plan "$D/N1-plan-v1.json" --sha256 202b21384d71bb7fc1747bebe206eb76d72f9aed28d7f9588fe6ec854eddf42d --evidence-dir "$D/N1-launch-v1" --submit
```

At this checkpoint D1 is complete167/EOF/full exact IDs, result SHA b444b984c710226af07c9b21eb3759ec09ca73e0192a19a258cf349c253fa209. N1 is already in preflight/submission, not yet claimed complete. Task total34/candidate031 total12 actual completed requests. P model root960270/UID500/start47632926; D894247/UID500/start47641845. These are epoch evidence, not permanently live facts; actual PID/READY/deadline checks remain device-side.

Next stage: collect N1 under a NEW output filename, require exact176/EOF/calibration and both roles4rank exports, freeze CPU comparison with capture39-D1-first-stage.json and new N1 receipt, then check/render/submit. No production patch before earliest actual sampled difference is known. Quant buffers may include padding; equal normalized BF16 does not establish all scales/expert rows/full-batch equality.

Both requests completed
-----------------------

N1 completed176/EOF/exact reference IDs/profile, result SHA17f27e07e75da15df34be5396ce823ca04cf26393974ad4d91531b40a2fe4033. All4rank exports per role complete. Task total35/candidate031 total13 real completed requests. Normal and Drestore use the same model epochs.

CPU preflight v1 rejected the basename `CPU-compare-plan-v1.json`: its existing device SHA4107e319c1d2de31f87fad2609d8c48b40bdf31f416d4215e3d3cedf840505d8 belongs to capture38. No controller, analysis or RPC started; no file was overwritten and no model restarted. Use batch-qualified basenames for global device control files.

The actual v2 CPU plan is `Capture39-CPU-compare-plan-v2.json`, SHA7a5acbbd2475407acef76bcf1bf6291cb72a2a152d85c08a8f5e71a973c4027a. It binds the same completed captures and unchanged comparison script. Submitted exactly once via:

```sh
python3 -B "$E/capture39_compare_entry.py" freeze --source '/Users/linshuo/项目/decode-remote/deployment/reports/publication-precision-20261101/capture39-D1-first-stage.json' --normal '/Users/linshuo/项目/decode-remote/deployment/reports/publication-precision-20261101/capture39-N1-first-stage.json' --plan "$D/Capture39-CPU-compare-plan-v2.json"
python3 -B "$E/capture39_compare_entry.py" submit --plan "$D/Capture39-CPU-compare-plan-v2.json" --sha256 7a5acbbd2475407acef76bcf1bf6291cb72a2a152d85c08a8f5e71a973c4027a --receipt "$D/CPU-compare-submit-v2.json"
```

Actual CPU result
------------------

Original controllerSUCCEEDED, report SHA4cb2bf229dc0529aa9016185c3bcc9123f515c4f5ba6b8e23f42e4dc710edda5. P7168 first unequal Block entry remainsL26 (6985/20480); D7169 first unequal Block entry remainsL26 (7233/20480). D L0..25 prepared normalized inputs/HC coefficients and actual attention inputs are byteequal where present. L25 actual shared quant buffers/present flags are byteequal; sampled parameter hashes afterEOF unchanged.

L25 attention output differs4583/5120BF16 (maxabs0.046875). Its subsequent seam post4HC differs4128/20480, FFN normalized input3923/5120, FFNout4859/5120; new FFN HC coefficients also differ. Thus the earliest observed unequal operation result in this profile is L25 attention output, before the FFN seam. No FFN patch is justified. This is not proof of complete earlier Q/KV/indexer equality.

Actual D attention topology metadata: L25 compress_ratio1, kv_source_layer_id20, index_source_layer_id24, n_heads64, TP1. Do not infer a different compression ratio from past experiments. Next gap: the Q and actual selected readset/contents consumed inside this L25 attention call, including the index-source24 / KV-source20 dependency. No model math root or whole-model repair is claimed.
