# SaccadeNet

56 小时数学建模黑客松项目：研究贝叶斯注视点网络在“大图找小细节”任务中的语义计算量与准确率。

**当前状态：实施中。已有合成画布、视网膜、训练权重、Level 2闭环、三基线和可续跑E1驱动；正式E1尚未运行，没有最终科研结论。**

## 从这里开始

| 文档 | 用途 |
|---|---|
| [详细执行计划](docs/superpowers/plans/2026-09-25-saccadenet-execution-plan.md) | 任务 ID、依赖、具体步骤、文件、验收、排期与退路 |
| [进度看板](docs/PROGRESS.md) | 当前完成度、进行项、阻塞和证据入口 |
| [交接说明](docs/HANDOFF.md) | 新接手者先看；真实状态与下一项动作 |
| [实验协议](docs/EXPERIMENT_PROTOCOL.md) | 数据拆分、公平对照、指标、成本、结果留痕 |
| [决策记录](docs/DECISIONS.md) | 默认选择、对原方案的补充和未决事项 |
| [V0.1 原始方案](docs/saccadenet_v0.1_summary.md) | 本轮计划的主要来源；其中数值预期尚未验证 |
| [V0 历史方案](docs/saccadenet_v0_summary.md) | 仅供追溯；与 V0.1 冲突时参考后者 |
| [协作约定](AGENTS.md) | 禁止删除、授权范围、Git 与文档更新规则 |

## 本轮确定的实施方向

P0：V0-lite + Level 2 的 E1（1080p / 4K / 8K / 16K）+ 全分辨率滑窗 / 一段降采样 / 粗到细两阶段三条基线 + 图 4、图 5 + 最小逐帧回放。

比较语义成本，也单列感知成本、总成本和实测时间。准确率不达预期、两阶段基线更强、16K 失败都应如实报告。P1 仅在 P0 收口后推进；视频与端到端微调属于 P2。

## 本地版本管理

实施工作树位于 `D:\saccadenet\.worktrees\implementation`，分支`feat/implementation`；主工作区`main`保留最初规划。未配置远端。查看记录：

```powershell
git status --short --branch
git log -5 --oneline
```

每个任务结束同时更新代码、详细计划勾选、进度看板和交接说明。权重、MNIST缓存和`runs/`被Git忽略，跨机器接手须另行复制；本地Git也不是异机备份。

在实施工作树使用`.venv\Scripts\python.exe -m pytest -q`验证代码，使用`-m saccadenet.exp.e1_resolution --config configs/e1.yaml --dry-run`查看正式实验矩阵。实际运行位置、权重hash、未完成项请先读[交接说明](docs/HANDOFF.md)。
