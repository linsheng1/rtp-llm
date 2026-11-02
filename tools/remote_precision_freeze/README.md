# Frozen entry for C3 precision experiments

`freeze_stage.py` promotes the existing side-session freezer, with a small adapter
for the current two-request four-rank capture. It never imports model code or
issues a request. Historical side-session examples remain offline examples.

The persistent execution entry is `launch_stage.py`. It executes **check → render
receiver/P/D → actual device CPU/batch checks → live role gates → existing
cluster23_dispatch.submit/device_controller_cluster23_v1**. It has no polling,
new supervisor, DSH process, or model restart logic. Request-once claims, failure
latches, finite child supervision, actual client prepare, empty-cache proof,
EOF/cache/ID checks, and four-rank barriers stay in the existing coordinator/peers.

A model candidate and this control-code commit are distinct identities. The
current model is `61887b83eb180619032a46f0f2dc0eb86248ee09`, using the existing
native incremental artifacts. This is an inner diagnostic, not immutable-wheel
acceptance. A successful component or capture is not whole-model correctness.

## New batch

Use absolute paths. Safe control receipts and source scripts live in the task
workspace; business input/IDs, native artifacts, weights and tensor files stay
on devices. `--bench` points to the existing task helpers. No business bytes are
accepted as upload assets.

```sh
ENTRY='/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze'
BATCH='/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/capture23'
python3 -B "$ENTRY/freeze_stage.py" freeze --recipe "$BATCH/recipe.json" --output "$BATCH/plan.json"
python3 -B "$ENTRY/freeze_stage.py" check --plan "$BATCH/plan.json" --sha256 PLAN_SHA --cli-help
python3 -B "$ENTRY/launch_stage.py" --plan "$BATCH/plan.json" --sha256 PLAN_SHA \
  --evidence-dir NEW_EVIDENCE_DIR --device-prepare --readiness
# After the original zero-request readiness controller succeeds:
python3 -B "$ENTRY/launch_stage.py" --plan "$BATCH/plan.json" --sha256 PLAN_SHA \
  --evidence-dir NEW_SUBMIT_EVIDENCE_DIR --device-prepare --submit
```

These are command templates. Existing recipe/plan/evidence files are write-once;
never rerun `freeze` against an existing output. The exact executed SHA, commands,
results and current device stages are in the task shared state and batch report.
The second command does not mean a device prepare happened. Only the coordinator's
actual `client-prepare` exit and outputs establish complete device preparation.

Source/candidate/input/instance changes require a new recipe and plan. Drift is a
hard failure; do not edit an old SHA. The original 45-second identity/heartbeat,
75-second fresh proof, finite RPC budget and runtime admission remain in force.
The launcher preserves the original 195-second per-phase active-lease check;
no lease is extended. Model preparation accounts for the complete lifetime and drain. PID/UID/start ticks,
READY and parent controller are rechecked on the device. The controller and client
repeat the relevant checks immediately before each phase/request.

`prepare_capture_models.py` only persists the existing compose/check and natural
lease-boundary handoff. It does not stop a healthy model. It creates fresh
observer nonces and uses the original after-drain wrapper and controller. Do not
replay an uncertain model launch: inspect its immutable receipt/status first.
`new_capture_recipe.py` binds its real CPU/dispatch receipts to the new recipe.
`bind_capture_controls.py` documents the minimal versioned control adaptation:
current helper hashes and roots, correct 175-token reference label, and complete
artifact probes before the short-lived empty proof. It preserves old scripts.

To run the 18 local regressions against the safe real receipts:

```sh
PRECISION_STAGE_RECIPE='/absolute/path/to/offline/stage/recipe.json' \
PRECISION_CAPTURE_RECIPE='/absolute/path/to/current/capture/recipe.json' \
python3 -B -m unittest discover -s "$ENTRY" -p 'test_*.py' -v
```

These tests check immutable publication, drift rejection, real CLI declarations,
role/input/observer/script binding, and old/new flow compatibility. They issue
zero model RPCs. Device tests and model results must be reported separately.

Collect a completed batch once (all three original controllers must be terminal):

```sh
python3 -B "$ENTRY/collect_capture_result.py" --plan "$BATCH/plan.json" --sha256 PLAN_SHA \
  --three-route-reference /absolute/path/to/root-dgdet-three-route-full-ID-actual.json \
  --output NEW_RESULT_JSON
```

The collector performs full ID-list equality on the seed device and exports only
safe metadata. It issues no RPC, changes no runtime and does not poll. Its own
source hash is recorded in the report separately from the experiment plan.
