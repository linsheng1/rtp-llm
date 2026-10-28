# Decode KV DRAM 第一批验证报告

更新时间：2026-10-28。目标分支：`alibaba/rtp-llm:feature/dsv41-dev-main-202161013`。

**状态：真实 DRAM 切分恢复及输出对照通过；复制组 D 已发布，但 native rank 1 在关停阶段发生 SIGABRT，复制恢复和安全退出未通过。旧 SDK 远端缓存配置在 P 就绪前启动失败；原无远端缓存模式的普通／PD 关闭态检查仍待完成。测试修正的单例及完整 52 项已实际通过。当前不能宣称整体验收完成或已合入上游。** 本文是本批唯一汇总报告，后续结果在本文追加更新。

## 交付和运行输入

- RTP 运行源码：`34ed7d7dad9468713401fe6671c1bc14c747d338`，tree `4970ded6e59f23f0f64b1599d5b6818bc9ad2585`。
- 代码候选：`695b3afd9034866372dac40d033722252a9f8881`，tree `84ac1f9423a0d31a35fa5be44fd97939aa2a2833`。与运行源码唯一差异为 `StreamCacheResourceTest.cc` 的测试快照顺序修正；生产代码、Python、proto 和默认依赖完全相同。后续文档提交不改变运行输入，不重新打包或替换正在使用的 SDK。
- 已通过普通 merge 纳入上游 `fad3ed5f1e5c65783aef4f5f927140114c9379e5`，保留双方全部历史。只追加提交，不 squash、rebase、amend 或强推。
- 现有 Manager 不变；第一批不以 Manager 来源元数据改造、SSD、GDR、性能扫描或 D 来源专属指标为前置条件。

| 用途 | SDK | 完整 wheel SHA256 | 实际结果 |
| --- | --- | --- | --- |
| 默认依赖编译和关闭兼容 | `2026_10_13_22_33-79404320` | `b28704ec0cd683721625f9b4d19c8bc62f9c32a9540d0c0caa3bd8ee9b5a3aa6` | 完整生产构建、独立安装与 CPU 加载/退出通过；远端缓存配置启动失败，无远端缓存原普通／PD 模型检查待完成 |
| 开启功能 | `2026_10_25_15_21-cuda130.aarch64` | `e447920282e59e7760db94756faafa3d1d17c30017598b53dd826680213e5a21` | 完整生产构建、私有运行安装及切分恢复对照通过；复制组待完成 |

配套 SDK 的编译头文件、链接输入和运行 DSO 来自同一 RPM，未使用旧头文件加新 DSO 的覆盖方式。功能构建仅对 ARM SDK 仓库输入使用显式版本补丁，派生输入 tree 为 `5cbee7872f2521b164b31961c49f561f6e61a2e6`；PR 中默认 SDK pin 保持上游值。

| SDK 身份 | SHA256 |
| --- | --- |
| 默认 SDK ARM RPM | `d4bfe547429c6f6d76f1278451a053e94bba570b7ecfc24c8eef152f749749a3` |
| 默认 SDK 运行 DSO | `1395c2c4bdfa1b7a42e4084a3dec4e79add9475a58c9bd2cd102dfcba468d72e` |
| 配套 SDK ARM RPM | `5b382202aa716a850984ce2acfb03981b05875b5b014cc8e9bc215dee22ddae4` |
| 配套 SDK 原始 DSO | `b7eb2d3592fc0c0340f0432e9013dbbd18ebdf6ee95542aaa6dc3edaca574300` |
| 配套 SDK Bazel 打包 DSO（含 SONAME/RPATH 处理） | `eccbafbcdc651c65df5c029f8f3326e954f1dbe601ed651b27c7fc69562d25ed` |

## 真实切分 DRAM：D 发布到 fresh P

使用真实模型、P CP=2 切分、D TP=1/DP=2 和现有 Manager/DRAM Provider。保留请求真实 token IDs，不用重新分词后的文本冒充相同历史；新 P 使用空本地缓存。没有以 GDR 为验收要求，也不宣称传输为 GDR。

| 场景 | P 已计算终点 | D 已计算终点 | D 已提交终点 | fresh P 本地/远端复用 | 输出核对 |
| --- | ---: | ---: | ---: | --- | --- |
| 较长生成的控制样本 | 1788 | 2811 | 2560 | 0 / 2560 | 两次冷恢复 512 个输出 IDs 全部一致；独立短 bypass 的前 32 个 IDs 与之相同 |
| 自然 EOS 边界样本 | 1790 | 1826 | 1792 | 0 / 1792 | 下一轮输入 1934；冷恢复与 bypass 的 32 个输出 IDs 全部一致，bypass 本地/远端均为 0 |

较长样本覆盖 D 新增逻辑区间 `[1788,2560)`，共 772 token；它达到生成长度上限，不标作自然 EOS。自然 EOS 样本输出 37 token，EOS 位于索引 36，最后采样的 EOS 未形成 KV；实际提交只覆盖 D 新增区间 `[1790,1792)`，共 2 token，未提交尾部 34 token。两种证据互补，不将自然 EOS 样本描述为覆盖一个完整的 D 新增 checkpoint。

自然 EOS 的初次短后续输入为 1852，只恢复到 1536。原因已由 Manager 实际六 key 查询及现有代码核对：模型 `cache_min_replay_tokens=128`，CP 复用单位为 256，`floor((1852-128)/256)*256=1536`。没有改生产代码或关闭该约束；保留原 D 数据，将下一轮真实用户输入增加到 1934 后恢复到 1792，尾部回放 142 token。中间 P 写入跳过全部七个已有 key，随后恢复仍读取原 D 发布的 URI。

D 成功依据为载荷完成及 Manager `FinishWrite` 确认后的提交记录，非“进入队列”或 retained 日志。记录显示应用层 `CPU_snapshot_bytes=0`；该记录不代表 SDK 内部完全没有主机暂存。

自然 EOS 原始 source P 两个 native rank、D 两个 native rank，以及最后 fresh P 两个 native rank 的原始 `waitpid` 均为正常退出 0；Consumer 为 0。独立复核按实际角色/rank PID 对齐原始子进程退出事件，没有用父进程 0 推断子进程状态。

测试补验的实际不可变输入树由 7972 条逐文件 manifest 独立重算为 `962d5b865951751e8cf1aac690fa900c382a8fce`，与代码候选 `695b3af` 仅差同一 ARM SDK 依赖变体。补验元数据从早先构建继承的 `5cbee787...` 是 `34ed7d7` 的派生树，属于过期描述字段，不作为补验输入身份；运行源码和整包身份没有变化。补验二进制 SHA256 为 `98f592cdb9fbadab50e02a64012ab7cddcb9ab37e0a7983bd4219a16f30526b7`。

## 原生回归与资源保护

默认 SDK 已完成生产 wheel 和目标二进制构建。最初原始记录共 18 个 XML、641 个 case，其中 17 个目标通过、一项 Stream oracle 失败；另一个 Prefill 测试因新 fixture 缺少 runtime metadata 崩溃。两处均通过追加提交修正，不抹去原始失败记录。

- Prefill 修正后的 54 case 通过；单独两 GPU writer 用例通过，补充此前单 GPU 运行跳过的用例。
- 原生发布完成专项 214 case 通过，覆盖页引用、预算、异步完成和失败保护；它使用真实 GPU 页和模拟 SDK，不能替代真实 Manager/模型闭环。
- 其余配置、预算、生命周期及 Python proto 检查已有通过记录。最终唯一 case 总数待 Stream 补验后按 XML 去重汇总，不把重复运行或跳过计作新增通过。
- `695b3af` 只将测试中的 key 快照移到异步分配开始前，保留物理 DEVICE 可用性、resident、无 HOST/DISK/REMOTE 复用及 backend 单次调用断言。该单目标重新编译已返回 0；配套 SDK1025 下精确失败用例复验及同目标全部 52 case 均实际通过，零失败、错误、跳过或禁用；这不是默认 SDK794 的全部原生目标重跑。

未知传输完成状态保留页、SDK 注册、载荷描述符和预算，并停止新发布；无传输排空证明时不能强行回收后报告正常退出。永久超时下的 SDK 排空和远端回收契约未完成端到端故障验收，不作保证。

## 剩余必需项（不扩展测试矩阵）

| 项目 | 状态 |
| --- | --- |
| DRAM 切分真实 D 发布、空 P 首次恢复、原 token 边界和短 bypass 对照 | 通过 |
| DRAM 复制一组基本闭环、native rank 退出与资源释放 | D 发布成功；关停 rank 1 SIGABRT，恢复及安全退出未通过 |
| 旧 SDK 准确候选整包、功能开关关闭的短模型兼容和退出 | 独立 CPU 安装、库身份及加载退出通过；真实模型待完成 |
| `695b3af` Stream 测试修正补验 | 编译通过；配套 SDK1025 下精确单例及完整 52 case 实际通过 |
| 汇总报告附入提交/上游 PR，普通 merge 到目标分支 | 报告已附入追加提交并推到 fork；上游 PR/merge 未完成 |

关闭态通过后才能记录“在本次实际用例中原功能正常”；未做性能对照，不能宣称零性能开销。历史 RC5 曾发生 D rank `SIGABRT`，当前切分样本全部 rank 正常退出不等于已证明修复历史根因，复制和关闭态仍需记录自己的真实退出结果。

SSD 优化和已有数据保留，性能问题未解决；SSD、GDR、D 来源专属指标和大范围性能扫描均未完成本批端到端验收，不作通过声明。重复请求仍有本地复用为 0 的观察，本批只记录真实冷恢复及对照，不声称本地热命中或性能收益。

## 复制关停失败及关闭态输入补充

复制样本源请求自然 EOS，P 已计算 1790、D 已计算 1826、提交终点 1792；17:30:32 已收到实际 DTOP 提交记录。17:32:41 两个 D native rank 同时收到正常停机信号并进入 `NormalEngine::stop()`，rank 0 正常退出；约 57 秒后 rank 1 出现 DeepGEMM `barrier.cuh:27` Timeout，随后在 `CachingHostAllocatorImpl::free` / `TensorImpl` 释放路径 terminate，17:33:39 发生 SIGABRT。原始 `waitpid` 明确记录 rank 1 为 signal 6；父启动器返回 0 不能覆盖该失败。尚未证明根因是发布功能或既有 DP 关停问题，也未完成复制模式 fresh P 恢复，因此不记为验收通过。已保全原始请求、日志和退出记录。原独立 DRAM Provider 随后按期限正常退出，实际载荷已失效；后续恢复验收需要按同一短 fixture 重建测试数据，不能把归档日志当作仍可读取的载荷。

默认 SDK 关闭态独立 runtime 已逐项校验完整 wheel 的 2147 个安装成员、固定 RPM 的五个头文件和三份实际加载的原生库，CPU 加载退出 0，CUDA 未初始化。后续构建已复用共享 Bazel 临时输出，不能拿该临时目录替代旧 wheel 的历史构建输入证明；本次依据固定哈希的归档 wheel 和原始 RPM 独立核对，没有覆盖 A 的 SDK 或共享输出。

## 默认 SDK 实际启动失败的范围

默认完整 wheel `b28704ec...`、SDK794、发布开关 `0` 的首次远端缓存配置在 P 就绪前失败，两 rank 报 `CUDA error: no kernel image is available for execution on the device`；没有发出普通 P 或 PD 测试请求。父进程退出 1，两个 native rank 由失败清理流程以信号 15 结束，不能记录为正常退出通过。后续 controller 的 runc 查询错误属于 namespace 退出后的次生错误。

只读 `cuobjdump` 核对发现：默认 SDK794 的实际 DSO 只含 `sm_103` cubin，无 PTX；配套 SDK1025 包含 `sm_100` 与 `sm_103` cubin。两包的主 RTP DSO 均有 sm100/103 镜像及 PTX，compute DSO 完全相同。这是 SDK 包能力差异证据；异步 CUDA 异常尚未锁定首个发起 kernel，不能据此直接认定 SDK、GDR 或新功能的因果。

下一项检查仍保留准确的原完整 wheel 和 SDK、发布开关 `0`，仅采用上游既有 `--enable_remote_cache 0` 的普通／PD 模式。它只能证明该无 KVCM 远端缓存配置下的关闭兼容性，不能替代旧 SDK 远端缓存路径的验收，也不会热换运行库或改冻结源码。当前尚无成功模型结果。

## 原始证据索引

原始日志、完整 IDs、XML、文件哈希和退出记录保存在本次工作区验证归档，本文仅列可用于核对的归档身份：

| 归档 | SHA256 |
| --- | --- |
| `batch1-A34-sdk1025-natural-EOS-minReplay-v6-raw.tar.gz` | `f206c821a06f2347dfd57cd39c5c9aefee5253b28f7be3d78946a14237a12e29` |
| `batch1-8e91fe6-native24-v1` 原始记录归档（含最初失败） | `147bd63e510339c1b040d10d8bef77d939dbe8085bf18c3cf7715cd1609905bf` |
| `root-695-stream-full52-raw.tar.gz`（52 项 XML、日志、库加载证明） | `d369e971980bd27ce4d541c099a83f2fd0da80e1684862d4dee362796aaa6657` |
| `695b3af` 相对运行源码的唯一测试补丁 | `7ce4128f0406bb23d13513c3d91dc9875e84db5fbe8fd72c2d83985216d92ca0` |

运行 wheel 身份不随测试或文档提交重新命名；上游最终提交需保持与上述运行输入的生产代码一致，或明确记录新的生产改动及所需补验。
