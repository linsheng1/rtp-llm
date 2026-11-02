# Candidate 031: actual runtime and frozen entry

Model candidate: `031fc2af33111e7bbe13bbca6e72ebd6fc094d7c`. This is an
incremental native/Python diagnostic, with no acceptance wheel built yet.
The existing four native-library hashes, SDK, Consumer and storage services are
retained. It runs only on C3 P `10.44.12.175`, D `10.44.61.71`, seed/storage
`10.44.61.73`. Root Codex is the sole executor; no DSH or delegated executor.

The production change prepares the same TF32 HC weight representation in the
large automatic DeepGEMM branch and the small explicit DeepGEMM branch. Its
separate six-call primitive proof is in `history36-HC-proof-safe.json`. The
actual candidate regression ran three tests, zero failures/errors/skips. One
test exercises the real CUDA delayed-HC M1/M513 boundary; two are existing CPU
TF32/cache contracts. Component success does not establish model correctness.

## Completed real requests

Candidate 031 source: G7041, output295, complete native EOF. Result SHA256
`5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa`.
Derived diagnostic input is the first7169 actual IDs of G+Y[:-1], SHA256
`3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c`.
Candidate 031 normal continuation: output176, native EOF, bypass local0/remote0,
result SHA256 `6c70f86ba92b085933ef371e17a9a380e622c90667e6c4eccae4283e88db9333`.

This input differs from candidate91b's derived input. These are actual-D-history
diagnostics, not the original business-R acceptance. Across this task, completed
requests at this checkpoint are618:8,91b:14,031:7, total29. The031 count includes
one deliberate P-only max1 publication seed. Three-route complete ID equality
is not achieved. Whole-model repair, business behavior, exact remote-byte
provenance and immutable-wheel acceptance remain unproven.

The new-instance D restore completed167 output tokens with local0/remote7168,
result SHA256 `506592ada04045010d963c0dda622a8da13266dc65ae245703c8175a8a30fc17`.
The same P/D epochs then completed normal176, result SHA256
`673144b2f28f89ffb1b9885045a61883552e2e34c64159354624f5c7f4e9a4b0`.
The CPU full-ID comparator proves their first difference at index32/token33;
result SHA256 `4eb97f0c23c1024e03eec2b0474a493ea8c72a22129d44100235732c3561120c`.
A second zero-RPC comparison proves the complete167 D-restore IDs exactly equal
actual sourceY[128:], and both normal epochs have identical complete176 IDs;
result SHA256 `e1695e8d292e005bd39215882b12e9f4f6e6e326a2751d8ec9c4e6c39b22de0b`.
These token facts establish neither a remote-byte join nor the remaining
operator cause. The model is still not fixed.

P-only seed completed one output token/EOF on the exact R7169 in a distinct P
publication namespace; native result SHA256
`223210f3cc153908b04a1dbc3038cb9b9ddcb27da1c1c764b66fb6fc878abcae`.
The original pre-seed Manager cursor and metadata collector verified14 new
canonical keys and14 readable locations, report SHA256
`9c422cfe82747d28ff32cf465fe29547ebf148d3e6f9d7aa010cd1af9ad57ded`.
P-restore completed176 tokens/EOF with local0/remote7168. A third normal
request on those same P/D epochs completed176 tokens/EOF. Complete-ID comparison
proves P-restore equals normal exactly, and D-restore differs first at token33.
All actual generation/route profiles match. Empty-tree observations were
54.686s (D restore) and19.974s (P restore) old at native request start, within75s.
The three-route safe report is `hc031-three-routes-safe.json`, result SHA256
`1a14fe93c362ff7ba7a7b8eae464714b583817d94c86c6bb736a426ce0e77635`.
Normal complete IDs agree across all three normal epochs. The overall model is
still not fixed; the remaining arithmetic point needs current-candidate evidence. The completed-seed/metadata proofs gate an owned P drain
through the original `handoff_state_epoch.py`; no Consumer or D is stopped.
The D2 lease cannot cover both measured P transitions plus the unchanged180s
RPC admission. D3 started only after the original successful D2 drain. The retained Consumer,
SDK, Manager and Provider were reused. Both P-restore and normal3 used that
same D3 model epoch; all epoch identities are in the safe comparison report.

The first D restore plan, SHA256
`64454fdefa489b4b19ad208b7c44a0dc51f0590e21cfac20ff375ff82c38a1e5`,
passed offline and actual device CPU preflight but issued zero RPCs: the original
fresh-P observer found `OBSERVED_NONEMPTY_LOCAL_TREE` after the prior requests.
The source1 models then entered their original lease drain. The next epoch uses
the unchanged original1800s lifetime,420s drain reserve and after-drain wrapper.
Its first request is remote restore; bypass normal follows it, avoiding a normal
request populating the fresh allocator before the restore gate.

Metadata v2 completed both controllers with exit0 and zero model requests:
native evidence SHA256 `516f5a649f15ed56923b00e492273a5d2e84b123949f01f03e7fc79f27c5c918`,
Manager report SHA256 `41b7f64ea44969402f61cb3f0865e39917341376a644571a8c6eb2cc882f1eaf`.
This is posthoc native-PID/Manager metadata, not a pre-request cursor or exact
source/remote/destination byte proof. The wrong-source-SHA metadata v1 and its
failed submission are retained in the workspace as rejected history; they are
not the active metadata entry. SSH255 is not treated as permission to replay:
actual device config/exit evidence was inspected and completed CPU checks reused.

## Persistent commands

All paths below are persistent. The source/recipe/dispatch receipts and dynamic
state are in `/Users/linshuo/项目/decode-remote/deployment/`; raw IDs, tensors,
models and weights stay on C3. No artifact bytes pass through the Mac.

```sh
ENTRY='/Users/linshuo/项目/decode-remote/deployment/branches/publication-precision-20261101/tools/remote_precision_freeze'
BATCH='/Users/linshuo/项目/decode-remote/deployment/coordination/runtime-freeze-batches-20261102/hc-candidate031'
```

For a **new** recipe/plan, use:

```sh
python3 -B "$ENTRY/freeze_hc031_routes_v3.py" --recipe NEW_RECIPE --output NEW_PLAN
python3 -B "$ENTRY/freeze_stage.py" check --plan NEW_PLAN --sha256 PLAN_SHA --cli-help
python3 -B "$ENTRY/freeze_stage.py" render --plan NEW_PLAN --sha256 PLAN_SHA --role receiver
python3 -B "$ENTRY/freeze_stage.py" render --plan NEW_PLAN --sha256 PLAN_SHA --role P
python3 -B "$ENTRY/freeze_stage.py" render --plan NEW_PLAN --sha256 PLAN_SHA --role D
python3 -B "$ENTRY/launch_hc031_routes_v3.py" --plan NEW_PLAN --sha256 PLAN_SHA \
  --evidence-dir NEW_LAUNCH_DIR --submit
```

The v3 launcher groups small immutable source publications below the original
100KB control limit, then runs CLI checks only after all groups are published.
The first P-seed v2 attempt had zero RPCs and no receiver spec because it checked
CLI before a new receiver in a later group existed. Original plans remain
unchanged; v3 reused the healthy model and completed the seed. It retains all byte equality checks, actual CLI prepare,
original submit/controller, live artifact probes,75s empty proof, request-once,
RPC lease and actual cache-result gates. It introduces no new supervisor.
Existing plans pin their original launchers and must never be updated in place.

The actual new D-restore batch is `HC031-drestore-plan-v2.json`, SHA256
`7cf0233e24fb84f0c05aa3cab20c1377ed5cf6c4c4a264839e065e4d22763903`.
Its P/D model epoch plans are respectively
`6579391a31df9573e1adb95f6298400261e2433ca587885f73f08c116869ce6f` and
`ba118e0ef8057fab9440e2be984a692525f55dbbbeb8c21ee0d77ff5cf44e2c8`.
The new normal plan is `HC031-normal-plan-v3.json`, SHA256
`fd6e4cf6049abe929b4f00bec108a6f2c5698a6c1ca39401f1c38e18e2027658`;
it completed after the D-restore stage ended. P-seed used
`HC031-Pseed2-plan.json`, SHA256
`dba86e2550c3da47713a0ee322e325d79e07eded08f509bdaa054437f2d73485`.
The completed P-restore plan is `HC031-Prestore1-plan.json`, SHA256
`5af9509dc69fbdcc71a5dd0053c1fa5f30d1c3bbcb7c78650d42562b6ff75e7f`;
its D3 model started after the original D2 drain and completed both the
P-restore and same-epoch normal3 requests. Normal3 plan SHA256 is
`5dc33fc06edc1a00dd12f328cec9b07f9105408c3050cd59db620a07f19be2d2`.

Read a phase once with a **new** output path; never replay a launch to query it:

```sh
python3 -B "$ENTRY/read_hc031_stage.py" \
  --plan "$BATCH/HC031-drestore-plan-v2.json" \
  --sha256 7cf0233e24fb84f0c05aa3cab20c1377ed5cf6c4c4a264839e065e4d22763903 \
  --output NEW_SAFE_RECEIPT --models
```

New model epochs use `prepare_hc031_epoch.py prepare --recipe NEW_RECIPE
--directory NEW_DIR`, followed by `check`, `render`, `submit` with the emitted
plan and SHA. The original wrapper verifies the prior successful owned drain,
zero remaining model processes and actual runtime admission before launching.
Completed exact CPU configuration may be reused only after plan/exit/probe
identity checks. Peer roles can be omitted from a recipe to preserve them.

Complete-ID comparison uses `compare_hc031_results_entry_v3.py freeze/check/render/submit`
with actual safe EOF receipts (`--normal`, `--d-result`, optional `--p-result`),
new `--plan`, `--root`, and existing `--bench`. It invokes the existing CPU-only
diagnostic comparator on the seed device, exports counts/hashes/first differing
positions and role epochs, and issues zero model RPCs. Comparison success is
separate from output equality and formal acceptance.

Latest dynamic stage, exact commands and follow-up gaps remain in
`/Users/linshuo/项目/decode-remote/deployment/state/publication-precision-20261101.json`,
key `HC_candidate031_20261103`. PID/lease/READY and cache proofs must always be
checked live; this document never makes a past snapshot reusable as admission.

Actual three-route comparison plan: `HC031-three-routes-compare-plan-v3.json`,
SHA256 `09972885857043b25a01a4b4d0a8f7e0237947a3137fb72b566668b17e263ec0`.
This CPU-only job completed with exit0 and checked complete IDs, exact profiles,
actual reuse lengths, historical empty-proof age and matching P epochs.
