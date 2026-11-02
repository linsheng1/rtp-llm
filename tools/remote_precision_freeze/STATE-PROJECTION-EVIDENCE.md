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
