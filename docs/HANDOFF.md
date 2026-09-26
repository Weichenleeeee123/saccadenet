# 接手说明

## 最新交接（2026-09-26 14:10，Claude，额度将尽）

- 本轮已完成：图0结构图（`reports/figures/architecture-20260926T053411952920Z/`，并已写入报告）；报告新增“什么时候注视是最优的（解析分析）”一节，路演第8页已同步。
- 进行中：工程优化，已登记为D49，代码**还没改**。剖析脚本在`.tmp/profile_episode.py`（运行时需`PYTHONPATH=.`）。实施顺序和验收标准见D49：先写等价测试，再改采样、检测、`_index`，逐局比对trace一致后，再在D48的画布上重测时间。
- 之后：用户想讨论视频追踪，即打破“目标位置无先验”这条前提的场景。
- 分支`codex/audit-v02`在本提交之后领先`main`，需要时再合并、推送。

## 上一次交接（2026-09-26 13:45，Claude接续Codex）

- **位置**：2026-09-26 13:55，用户要求合并并推送：`codex/audit-v02`（到`00afa54`为止）已通过本次合并提交并入`main`，并推送到`origin/main`（GitHub `Weichenleeeee123/saccadenet`）。主工作区`D:\saccadenet`现为`main`；实施工作树`D:\saccadenet\.worktrees\implementation`仍在`codex/audit-v02`，内容与`main`一致。本地标签`v0.1-demo`、`v0.2-analysis`以及两个开发分支都没有推送。
- **做了什么**：Codex在审计A01之后完成了B01–B04（计时修正、强两阶段、失败诊断、定位修复、确认规则负结果、路由和概览宽度对照），并冻结了新留出评估D47；在第864/1200局时其额度耗尽，进程终止。Claude在同一commit、干净工作树上续跑补齐到1200/1200，写了`saccadenet/exp/analyze_holdout.py`，记录D48，更新报告（新增“修复与新留出评估”一节，改正A01指出的旧表述）和路演提纲，勾选了复核计划B04。
- **结论**（D48）：新画布上三种方法都是400/400，同画布没有不一致的配对（天花板）；语义算量持平；SaccadeNet每档慢4.2–6.5倍。在本任务上，它在所测维度没有一项优于强两阶段。
- **没有运行中的进程**。98项测试通过。磁盘（13:40）：C约0.8 GB，D约3.9 GB。16K及以下的运行没有再撑大页面文件；大于16K的运行不要做。
- **下一步**：①人工：确认赛事格式，按提纲做幻灯片，排练和提交；②可选实现优化：每眼的Python开销（1080p/4K眼数相近仍慢约4倍），不改变方法结论；③未做的鲁棒性项：真实视网膜重建增强（复核计划B03第一项）、无卡片任务。④Demo仍用修复前的E1 trace，路演用它展示“失败与修复”时要这样说明。

## 合并交接（2026-09-26 12:23，Codex，历史记录）

（这一段写于第一次合并`f63f364`之后，当时尚未配置远端；现在的状态以上面“最新交接”为准。）

主工作区 `D:\saccadenet` 现为 `main`，已通过合并提交 `f63f364` 纳入 `codex/audit-v02` 的已提交工作（审计、计时修正、强两阶段基线、诊断与定位修复）。实施工作树 `D:\saccadenet\.worktrees\implementation` 仍保留在 `codex/audit-v02`，没有删除或清理文件。主工作区原有 `.claude/` 仍存在，现由 `.gitignore` 忽略。没有远端，也没有推送。

合并后在主工作区运行 `D:\saccadenet\.worktrees\implementation\.venv\Scripts\python.exe -m pytest -q`：87 passed、1 条第三方 `dateutil` 弃用警告。当前没有运行中的实验进程。本次按用户要求暂停后续实验；下一项为恢复计划中的 B04 与最终确认，须从 [恢复计划](superpowers/plans/2026-09-26-saccadenet-recovery.md) 和 [B03 报告](../reports/recovery/b03-localization-20260926.md) 接续。B03 的开发集定位改善不等于最终验证，现有速度仍落后于强两阶段基线。

## 历史审计交接（2026-09-26，Codex）

当前工作树仍为 `D:\saccadenet\.worktrees\implementation`，分支已切到 `codex/audit-v02`；审计起点 `2846d58`，历史实施分支 `feat/implementation` 保留。本轮仅审计代码/已有 CSV 并更新文档，没有训练或新推理。最新提交用 `git log -1 --oneline` 查询，标题 `docs(A01): audit v0.2 claims and prioritize performance recovery`。

先读 [AUDIT-2026-09-26.md](AUDIT-2026-09-26.md)：准确率复核 390/400 对 399/400；识别训练/定位/融合未被排除为失败因素；计时混入 FLOPs 计数前向，字节为逻辑估计；E2 斜率只拟合 K≤16 / K≤32。旧报告未改写，阅读时须结合审计中的限定。下一步优先测量与强基线、定位/重建/校准，再考虑搜索和架构对照；建议仍未执行。遵守当前会话授权，不把这些建议当作新增实验授权。

本轮没有启动后台任务；未删除任何文件，未改分页设置。磁盘实查 C 约1.48 GB、D 约3.46 GB。主工作区 `D:\saccadenet` 仍是规划分支并有原有未跟踪 `.claude/`，不要误在其中运行实验。完成状态仅在详细计划 A01 打勾，下方为此前交接记录。

更新时间：2026-09-26 10:10（Asia/Taipei）。本次交接点：P0已收口（标签`v0.1-demo`）。之后完成了P1：E2集合大小（D29/D30）、扇区检测探针（D31/D32），Demo加入了24K对照并生成备用截图，打标签`v0.2-analysis`。剩下的限时排练和正式提交需要人来完成。**当前没有运行中的实验进程。**

## 现在实际有什么

- 主仓库：`D:\saccadenet`；实施工作树：`D:\saccadenet\.worktrees\implementation`，分支`feat/implementation`。从工作树继续开发，主工作区`main`保留规划。
- 原始文档：`docs/saccadenet_v0_summary.md`、`docs/saccadenet_v0.1_summary.md`，原样保留；快照提交 `5322323`。
- 新增规划文档见根目录 README。规划提交可通过 `git log -5 --oneline` 查看，标题为 `docs: add execution plan and handoff tracking`。
- 正式E1：`runs/20260925T152349319841Z-a6a00495-e1-final`（a6a0049，clean，1,520/1,520）。结果与判定见`docs/report.md`，审计见`reports/e1/qa.md`，图4/5见`reports/e1/…-20260925T153721798894Z/`。
- D24缩图公平性补跑：`runs/20260925T155609286059Z-71b803aa-g2v2-final`；图5b见`reports/e1/g2-sensitivity-20260925T160936102578Z/`。D26 trace分析（图6）见`reports/analysis/trace-analysis-…161032135437Z/`；解析模型代码在`saccadenet/retina/horizon.py`。
- E5（D27/D28，24K外推检验）：`runs/20260925T161450886661Z-a4660741-e5-final`（a466074，clean，150/150局，已完成）。`…161406097212Z-7ff5cbd4-e5-final`是因dirty而停掉的那次，只有1局，只作审计，不参与分析。最终图6/7/8所在目录列在D28里。
- E2（16K、K=4–64）的run目录列在D30里；K=8请用clean的`…012810581470Z-cdf49baf-e2k8-final`，dirty的`…011914091930Z-867db6c1-e2k8-final`只作审计。图9在`reports/analysis/setsize-20260926T013124432780Z/`，图10在`reports/analysis/horizon-probe-20260926T012629819570Z/`。
- 最新Demo是`artifacts/demo/demo-20260926T013441565773Z/index.html`，共4局：16K、24K冻结、24K推导、1080p失败；备用截图在`reports/replay/snapshots-20260926T013503985083Z/`。
- 报告`docs/report.md`；路演提纲`reports/presentation-outline.md`；复现命令与独立复现记录`docs/REPRODUCE.md`；P0覆盖核查在`docs/PROGRESS.md`。
- 复现用的独立工作树`D:\saccadenet\.worktrees\repro-20260926`（detached，a0b39ec）只用于核对，不在里面开发。里面有复制来的`data/`、v1权重、一次冒烟run和`.tmp/`，以及一个未跟踪的`.tmp_fig.txt`。这些都可以由用户决定是否清理，代理不删。
- Demo：`python -m saccadenet.viz.demo_player --run <E1 run> --episode method:WxH:seed=说明`生成到`artifacts/demo/demo-*/index.html`，可以直接双击离线打开；清单在`reports/replay/`。
- MNIST在忽略目录`data/mnist`。
- v1主权重：`checkpoints/fovea/20260925T145236155331Z-b107b36e/epoch-008.pt`，SHA256 `3d80b18c01e0cb25fb3cc01dea891a5a5f1dcb35bbd8676102e490718bf8eda9`；retina标定：`reports/calibration/calibration-20260925T145532233348Z-3d80b18c-fit.json`；两阶段Platt：`reports/calibration/two-stage-platt-20260925T151159460820Z-3d80b18c.json`。权重被Git忽略，跨机交接必须实际复制。
- 开发32局冒烟首次全部因trace序列化报错，错误保留于`runs/20260925T151457474939Z-d222f1d5-e1-smoke`；修复后同目录第二次attempt 32/32成功。随后新增`sensing_bytes`字段，旧目录schema不可续跑。最终schema另建`runs/20260925T151923188489Z-d222f1d5-e1-smoke`，32/32成功、`summary-2.csv`有理想滑窗成本列。
- 没有远端；不能认为项目已备份到 GitHub。
- 已核验：RTX 4070 Laptop / 8188 MiB VRAM，驱动580.88，RAM约15.7 GiB；2026-09-26 00:17 D盘剩余约5.2 GiB，C盘一度剩0字节：系统管理的`C:\pagefile.sys`在24K实验时扩到了2.6 GB。09:13复查时C盘剩1.6 GB，D盘剩4.5 GB。如果再跑大于16K的实验，C盘可能又被占满，需要用户重启，或把C盘分页文件关掉（D盘已有24 GB分页文件）。这是系统设置，代理不能改。大于16K的实验会推高内存，不要并行跑。项目venv中PyTorch 2.6.0+cu126已通过GPU小张量测试。16K全分辨率滑窗开发首局推理约6.8秒，画布生成约4.6秒，尚无需租算力。

## 接手后的第一步

```powershell
Set-Location -LiteralPath 'D:\saccadenet'
git status --short --branch
git log -5 --oneline
git remote -v
```

随后进入实施工作树，阅读最新`AGENTS.md`、`docs/PROGRESS.md`、详细计划与实验协议。用户已明确授权持续实施。

H0记录为2026-09-25 22:13 +08。用户于09-26 00:00确认最终只交SaccadeNet，队友在休息；截止按2026-09-27 12:00排期，依据是同赛道记录，用户没有异议。

## 必须记住的风险

1. Level 2 不能读取真值候选位置；原方案主循环的 `meta` 参数需要拆分，真值只在生成、离线训练监督和评测中使用。
2. 从视网膜反投影候选视图，禁止按候选额外读取金字塔或高清原图。
3. 只有一个已发现候选时 softmax 为 1，并不等于全局 100% 把握；详见决策 D04。候选为空时也必须有合法探索策略。
4. 融合 B 按观测质量选证据，不能取证据数值最大值。模型 B 的高斯嵌套假设尚未验证。
5. 16K RGB uint8 原图约 398 MB；float32 原图约 1.59 GB，全金字塔与中间张量会进一步放大。必须先测内存，并在仅余约9.7 GiB的D盘安装环境前确认存储位置；绝不自动清理。
6. 三条 P0 基线必须齐全；已知候选基线仅用于 Level 1。
7. ≥98%、d′>1、语义成本≤4倍、准确率下降≤3个百分点都是待检验门槛，不是结果。

## 本轮设计补充

`docs/DECISIONS.md` 记录了种子拆分、单候选停机保护、缺失候选探索、检测匹配口径和成本拆分等默认提案。实施中先用开发集验证，冻结后再跑最终测试；修改必须追加记录。

## 每次离开前更新这里

- 当前分支、最后完成任务、相关提交。
- 已验证的命令、输出和失败原因。
- 当前运行目录、进程/会话标识、预计结束时间；没有则明确写“无”。
- 新文件/未提交改动的归属；禁止删除或覆盖他人工作。
- 下一项唯一优先动作、阻塞条件及解除方式。
- 复现需要的配置、权重清单、数据版本和产物位置。

当前下一项（按优先级）：
1. **人工**：拿到赛事的提交格式、页数和时长，按`reports/presentation-outline.md`做幻灯片（图和截图都已就绪），至少限时排练两次，然后正式提交并保存回执（T15最后一项、T16最后两项）。
2. **可选**：卡片关闭的E2第二部分；T17缩减版E3。按模型分析，这两项对主结论的边际价值较低，时间紧可以不做。
3. 如果改了代码：新提交，重跑受影响的测试和冒烟，不覆盖已有标签，另打新标签。**多个run串行时，运行期间不要在仓库里新建或修改文件**，否则后启动的run会被记成dirty；草稿写到`.tmp/`。

未完成且如实保持未勾选的验收：T06真实视网膜重建增强与过拟合检查、T07无效mask计数与高斯拟合诊断、T09 Level 1诊断入口。查看`git status --short --branch`确认有无未提交改动；`runs/`、`artifacts/`被Git忽略属于正常，不是丢失。

（2026-09-25的旧下一项“运行正式E1”和2026-09-26 00:22的“T15/T16”都已完成，见上。）
