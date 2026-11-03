当前冻结诊断输入的精度修复已通过完整 wheel 的真实模型验证。正常、P→remote→P、D→remote→P 的完整 166 个 token IDs 逐个一致，首个差异均为无；在最终相同 P/D 实例上的追加正常请求也一致。设备端还比较了完整输入列表、实际生成参数、实际源输出剩余序列、缓存与角色实例证据。

正常对照的准确含义：P/D 部署仍启用 remote 功能及 D publication；baseline 请求使用 bypass，绕过缓存复用，实际本地命中0、remote命中0。本次没有另行部署完全关闭 remote 开关的普通 PD，也没有证明与该独立配置的输出一致。上面的三场景一致结论仅对应同一候选、当前输入和这组实际配置。

源码候选：`fb09988a5c552f4f5ff716472d8b50647af22d2c`。已执行的持久控制入口 commit：`989caed99c74492da82f59d1355b2e7c1c0ce097`。本报告及 [安全证据](indexer-head-fullwheel-final-evidence-safe.json) 绑定这两个身份；报告归档 commit 只追加证据，不改变运行代码。

完整 wheel 仅在第三套构建一次，1922266511 字节，SHA256：
`6a77459e273388b3e6143b9b2c45d16e083cc6c4d658d382438e5ed5abfb4645`。
设备路径：`/home/admin/linshuo.zjl/publication-precision-20261101/control/headfb09988-full-wheel-v1/published/rtp_llm-0.2.0-cp310-cp310-linux_aarch64.whl`。
发布清单 SHA256：`ef9b4e3947ee7fd5bed45e3d5fc51118e64d1cc9bab6d56eab6d810a94a790a1`。
P/D 在设备内直接下载、逐成员核验并安装新私有包视图；Torch、DeepGEMM、Consumer、权重租约和原控制器复用。

| 实际请求 | 输入数 | 输出数 | 终态 | 本地 / remote 命中 |
|---|---:|---:|---|---|
| D 来源生成 | 7041 | 294 | 完整 EOF，未达 1024 上限 | 0 / 0 |
| 初次正常 | 7169 | 166 | 完整 EOF，未达 1024 上限 | 0 / 0 |
| D→remote→P | 7169 | 166 | 完整 EOF，未达 1024 上限 | 0 / 7168 |
| P-only 种子准备 | 7169 | 1 | max1 到上限后 EOF | 0 / 0 |
| P→remote→P | 7169 | 166 | 完整 EOF，未达 1024 上限 | 0 / 7168 |
| 最终实例正常回归 | 7169 | 166 | 完整 EOF，未达 1024 上限 | 0 / 0 |

本轮整包共完成 6 条生成请求，任务累计 57 条。5 条自行返回完整 EOF 且未触及上限；另 1 条为明确的 max1 准备，非自然业务生成。完整 EOF、未触及上限及没有外部中断有回执，但原 native 协议没有权威 finish-reason 枚举，本批也没有绑定运行时 ModelConfig EOS 配置，不将其表述为已验证 EOS1 终止。READY、心跳或仅控制器退出没有计作生成完成。

R7169 的 SHA256：`10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71`。其内容为本候选实际 G7041/Y294 历史的相同前缀。三场景及回归的完整 166 输出 SHA256 均为 `c8f32290469776961c8c562f05cb4924876ffbdbf258f1a4dc8dcfc640b103ca`。实际生成参数 SHA256 为 `8b08c68b5accf3d411deab7495aff702d484c00631a4668393367ae92dcb4cee`，max_new_tokens=1024、random_seed=42；未加入最小长度或忽略 EOS 参数。原始 IDs、protobuf、张量和制品字节均留在设备。

最后一个已证实的计算差异位于 L24 indexer 头权重投影。固定相同采样行与 checkpoint 权重，M4/M1762 的 BF16 F.linear 对照分别复现实际两侧的 32 个头权重。这两个试验批次的周边行是复制构造；原 P/CED 完整批次及实际 M 未补采，不能据此声称它们已相同。第 11 头的真实点积接近 BF16 舍入中点；仅替换该头便能复现两侧全部 7046 个实际分数，继而改变 GLOBAL TopK 成员。两侧 TopK 都遵守相同稳定 tie 规则。已有 L0 attention mHC split16/1 及后续定位证据保留，本轮没有重跑旧算子实验。

最小生产修改：受支持的 SM10/BF16/32×5120 头投影使用固定逐行、逐头 FP64 树归约，再一次舍入为原 BF16 输出。P 投影入口和 D prepared-weight 公共边界使用相同实现，Q/尺度、原 overlap 依赖及 TopK 规则保持；其他形状和训练路径保留 fallback。[内核](../../rtp_llm/models_py/modules/dsv41/fp8/_v41_indexer_head_projection.py) SHA256 为 `66851b81c1ee5567fcccf68aa087480fc1d4dc8fbe4b84ebe2897fd2cab957e7`。已有实际采样行的两种批形 FP64 参考、CUDA Graph 更换输入回归通过；包内代码逐字节相同，复用该证据。

整包门禁使用同一不可变 CP310/aarch64 wheel。原 SDK RPM SHA256 为 `5b382202aa716a850984ce2acfb03981b05875b5b014cc8e9bc215dee22ddae4`，五个实际生成头与运行 DSO 来自同一 RPM。NVML 实测 SM103，容器 glibc2.32、Torch2.11.0+cu130、NVCC13.2.86、fatbin10.0/10.3；RECORD、ELF、源码、protobuf 和标签核验通过。四个 native 库与已验证内环完全一致，1072 个共有生产 Python 文件也一致。

完整构建重新生成的 protobuf 仅新增三个 MemoryOperationRequestPB 追踪字段，原字段、服务和枚举不变。v1 检查因此在实际 RPC0 阻塞；v2 准确固定新成员 SHA，复用完整下载和提取结果，没有重建环境、重传或覆盖旧 plan。

持久接入：`freeze_indexer_head_fullwheel_routes.py` → `freeze_stage_fullwheel.py check/render` → `launch_indexer_head_fullwheel_routes.py` 完整 CPU 准备 → `submit_indexer_head_fullwheel_prepared_routes.py` → 原 `cluster23_dispatch.submit/device_controller_cluster23_v1.py`。构建、安装、源请求和准确恢复命令见 [持久入口](INDEXER-HEAD-RUNTIME.md)。以下真实批次均有离线 check、真实 CLI、引用发布/复用、设备完整 prepare、原 RPC 准入、request-once 和终态回执：

| 批次 | plan SHA256 |
|---|---|
| 整包来源请求 | `66672d591854b629a0a4b5ff4c4f1ba8c901d318ed24e10fb8ccd648e8f518be` |
| D→remote→P | `1cca041b404dda6f21ca086f65637c3b8b586166365694adf3b2808b6287af86` |
| P→remote→P | `680c1b3d6d5e47e8c4684440337e016d542c9a394292335bcfe14bc95e5748c5` |
| 最终正常回归 | `ea0dc322433a8284b2c3f08fc341faee73975b6bbbfaadc4d4dd06f4beabe070` |
| 设备完整比较 | `cac8518652337369e73ac7bff6888ab4349f611a0ac3cdf8403f38aca39bd276` |

比较结果：`control/headwheel-three-routes-compare-v1/result-safe.json`，SHA256 `88420ef443fbd7261fb129f88aa0265ca2c11487890cc6477d58e799d386c84c`。D/P 两条 restore 请求前本地空树均为 0 节点、observer RPC0，新鲜度分别 2.417988s / 1.730408s，满足原 75s。P 种子 Manager 回执为 14 个 key、14 个新 location、pending0。remote7168 是总读取命中，未全量标成 D 新增。

最终 P PID1136033/start51491197、D PID1046658/start51441894，均 UID500、四 rank，最后核验仍 READY。设备仅第三套 P12.175、D61.71、独立 Manager/Provider61.73。原有限服务总预算：P 至 19:02:08、D 至 18:53:53（2026-11-03 +08:00），原协议提前 420s 开始排空。Consumer7、完整内容核验 static keeper4 和设备本地监督保留，未续租或重启健康 D。

一次源 P 边界回执目录误撞 `<controller-name>-stop`，外层包装进程被取消。此前两条请求均已完整 EOF；旧模型根退出码未知，未伪造 final/rc0。设备证明全部旧 epoch 消失及 GPU 空闲后，新冻结边界 plan 直接复用原服务/控制器启动已 CPU 准备的 fresh P。新停止入口会在发布 plan 前拒绝该命名冲突，准确旧输入的 CPU 回归通过。随后各 P/D 排空均实际 rootrc0、cleanupFINISHED、remaining[]。旧失败证据保留，没有重复生成请求。

本次结论限于当前实际 D 历史 R7169 的完整模型数值门禁。原 5474/6988 业务 fixture、源/remote/目标完整 KV 字节来源验收、运行时 EOS 配置绑定和性能对照未纳入本次门禁；未声明整体功能/性能验收或零性能代价。当前数值门禁无剩余 token 差异。
