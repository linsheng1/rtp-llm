# Executed integration, 2026-11-02

The persistent entry now runs recipe → immutable freeze → check → render three
argv → existing submit/device controllers. The actual batch was `capture23`,
plan SHA256 `e9d7fc2c30a7c91080ae95ea00d52deed8bddf8141b85cb36c541fb1c98a6833`.
No side-session example was deployed. Failed earlier plans were retained.

18 offline/control regressions passed. All three actual CPU CLI/batch checks,
live role gates, actual coordinator client prepare, request-time fresh empty
proof, EOF/cache gates and both four-rank barriers passed. All three existing
controllers reached SUCCEEDED with command rc0. The candidate remained
`61887b83eb180619032a46f0f2dc0eb86248ee09`; native artifacts/SDK were unchanged.
The final helper sources are archived under `device/`, byte-identical to the
published task helpers. This control-code commit does not replace the model
candidate identity. No full wheel was built; this is a native inner diagnostic.

Two real requests completed (candidate cumulative eight). D-origin restore gave
128 tokens and normal bypass gave 175; both reached complete EOF below cap.
Their complete IDs exactly reproduced their prior route references. P-origin
restore remains exactly equal to normal. D-origin differs first at token33.
Runtime EOS binding was not independently verified. The whole precision repair
is still open. `integration-actual.json` records the actual immutable report,
role epochs, cache gates, result hashes and complete-ID comparison.

Minimal control fixes retain old helpers unchanged: current instance-root
validation, explicit current core/diagnostic/probe hashes, accurate normal175
labels, and both artifact probes before fresh-empty sampling. The bounded
READY-wait phase is separate from request submission. The 45s identity, 75s
empty proof, 195s active-phase admission and original failure/once gates remain.
CPU/import preparation runs under the actual seed environment only on seed;
P/D metadata preparation uses their own CPU-only interpreter environment.
Model transitions use original natural lease boundaries, with no healthy-model
restart. Initial handoff-budget failures are recorded in shared task state.

Read `README.md` and the task's shared state for exact commands and current
state. Never replay this completed batch or overwrite its plan. Read-only result
collection uses `collect_capture_result.py`; its SHA is recorded separately in
the collected report because it does not change or rerun the frozen experiment.
