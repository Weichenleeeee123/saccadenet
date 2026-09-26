# 复现与交接操作

实施根目录：`D:\saccadenet\.worktrees\implementation`，分支`feat/implementation`。正式E1从提交`a6a0049`启动；逐局manifest记录完整Git SHA、dirty状态和所有权重/标定哈希。`main`是原始规划工作区，不能误以为它包含实施代码。项目规定永不删除文件；重跑使用新目录，保留失败证据。

## 文件与环境

Windows PowerShell + Python3.12；开发机RTX 4070 Laptop 8188 MiB。`.venv`按[环境记录](environment.md)创建，本机venv继承系统科学库并覆盖PyTorch CUDA版，锁定版本在[`requirements-lock.txt`](../requirements-lock.txt)。跨机可建立新的独立venv并从官方PyTorch wheel源安装对应CUDA版本，但应按实际驱动重新验证兼容；不要假定复制venv目录可直接使用。

需要实际携带的本机忽略产物：`data/mnist/`、`checkpoints/fovea/20260925T145236155331Z-b107b36e/epoch-008.pt`、`runs/20260925T152349319841Z-a6a00495-e1-final/`、选定回放所在`artifacts/replay/`。Git仅含代码、配置、小型报告与图；未配置远端，本地提交不是异机备份。MNIST来源/文件哈希见[`candidate_recall.md`](../reports/spikes/candidate_recall.md)。

权重SHA256：`3d80b18c01e0cb25fb3cc01dea891a5a5f1dcb35bbd8676102e490718bf8eda9`；视网膜标定SHA256：`0fd620c48ce418b549a11d760eb83f2adc77b5923c410df734b53e7ff9c56281`；两阶段Platt SHA256：`2edaa935f786872dcac2c023698f96c1e14220017890e1ddfdf782e9b3e123f9`。完整路径和参数见[`configs/e1.yaml`](../configs/e1.yaml)。换权重、标定或配置时不能`--resume`旧run。

## 从代码到实验

在实施工作树执行：

```powershell
Set-Location -LiteralPath 'D:\saccadenet\.worktrees\implementation'
git status --short --branch
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e1.yaml --dry-run
```

Dry-run应列四档四方法、1,520个逻辑局、权重/标定SHA256。开发冒烟命令是`--smoke`，每档两局、共32局，另建run目录。正式运行与已建运行续跑命令：

```powershell
.\.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e1.yaml
.\.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e1.yaml --resume runs/20260925T152349319841Z-a6a00495-e1-final
```

正式run内的`manifest.json`有Git SHA/dirty、设备、版本与输入哈希；`config.json`是展开配置；`seeds.csv`含原始数字ID；`episodes.csv`逐局flush；`trace.jsonl`记录注视与后验；`errors.jsonl`保留失败。逻辑键为方法+分辨率+种子，哈希在manifest中固定。重试增加attempt，不覆盖原失败。若JSONL末行截断，新日志段另名追加，原文件不修剪。E1总结表在完成时生成`summary.csv`或编号版本；若中断在总结前，续跑可重新生成新编号总结。

## 图表与回放

正式run完整后：

```powershell
.\.venv\Scripts\python.exe -m saccadenet.viz.figures --run runs/20260925T152349319841Z-a6a00495-e1-final --out reports/e1
.\.venv\Scripts\python.exe -m saccadenet.viz.export_replay --run runs/20260925T152349319841Z-a6a00495-e1-final --episode saccadenet_lite:1920x1080:30000 --format png
```

制图器从完整`episodes.csv`读数，缺局会报错；每次输出唯一目录，包含图4/5 PNG/SVG、`plot-data.csv`、`provenance.json`。图4为语义FLOPs、bootstrap 95%区间；图5为目标命中率、Wilson 95%区间，失败计入分母。16K全分辨率n=20，其余方法/尺寸n=100。`plot-data.csv`同时列感知FLOPs、读字节、估计总FLOPs与墙钟。图不使用开发冒烟目录。

回放从保存的trace读决策和成本，仅用记录的seed重新生成相同画布与视网膜观察，不重新运行CNN/策略。绿圈真值只在最后一帧由评测层叠加；蓝叉是模型答案。每个frame对应一眼，`replay.json`写帧数、run和Git SHA。可另外对最终失败局导出，选择理由需在报告注明。

## 补充检验、分析图与离线Demo（2026-09-26追加）

下列命令都只读取已保存的run，或者新建run目录，不改动正式E1。每条的预测、判定和证据见`docs/DECISIONS.md`中的D24–D28。

```powershell
# D24：v2尺度增强权重重跑一段缩图（需要 checkpoints/fovea/20260925T145914395353Z-b43451ba/epoch-005.pt）
.\.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e1_g2_v2.yaml
.\.venv\Scripts\python.exe -m saccadenet.viz.figures --run runs/20260925T152349319841Z-a6a00495-e1-final --g2-supplement runs/20260925T155609286059Z-71b803aa-g2v2-final
# E5（D27）：24K外推，约25分钟；内存峰值约5 GB，不要并行其他大任务
.\.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e5_extrapolation.yaml
# D26：trace描述性分析（图6）、两个视野半径（图7）、标度律（图8）
.\.venv\Scripts\python.exe -m saccadenet.exp.trace_analysis --run runs/20260925T152349319841Z-a6a00495-e1-final
.\.venv\Scripts\python.exe -m saccadenet.exp.trace_analysis --run runs/20260925T161450886661Z-a4660741-e5-final --method saccadenet_derived_grid
.\.venv\Scripts\python.exe -m saccadenet.exp.trace_analysis --run runs/20260925T161450886661Z-a4660741-e5-final --method saccadenet_lite
.\.venv\Scripts\python.exe -m saccadenet.viz.horizons --calibration reports/calibration/calibration-20260925T145532233348Z-3d80b18c-fit.json --horizon <trace-analysis目录>/horizon.csv
.\.venv\Scripts\python.exe -m saccadenet.viz.scaling --e1-run <E1 run> --e5-run <E5 run> --e1-analysis <E1分析目录> --e5-derived-analysis <E5推导网格分析目录> --e5-frozen-analysis <E5冻结网格分析目录>
# 质量抽核
.\.venv\Scripts\python.exe -m scripts.qa_e1_crosscheck --run runs/20260925T152349319841Z-a6a00495-e1-final --plot reports/e1/20260925T152349319841Z-a6a00495-e1-final-20260925T153721798894Z
# 离线Demo（HTML，可双击打开；导出时自动校验帧数和成本）
.\.venv\Scripts\python.exe -m saccadenet.viz.demo_player --run runs/20260925T152349319841Z-a6a00495-e1-final --episode "saccadenet_lite:15360x8640:30000=16K 第一个测试种子" --episode "saccadenet_lite:1920x1080:30002=第一个失败局"
```

分析脚本每次都写入带时间戳的新目录，报告引用的最终目录列在D28里；同名前缀、时间更早的目录是排版草稿，保留备查。

## 独立工作目录复现记录（T16，2026-09-26 09:12）

在新建的独立工作树`D:\saccadenet\.worktrees\repro-20260926`上完成，detached，HEAD为`a0b39ec`。只复制了被Git忽略的`data/mnist/`和v1权重（SHA256已核对），复用同一个venv，没有删除或修改原目录。结果：

- `pytest -q`：79 passed。
- 开发冒烟`--smoke`：32/32成功，run为`runs/20260925T163142198555Z-a0b39ec5-e1-smoke`（在复现工作树内）。与原冒烟`runs/20260925T151923188489Z-d222f1d5-e1-smoke`逐局比对`target_hit/steps/semantic_flops/reason`，差异为0。
- 用正式E1保存的`episodes.csv`重画图4/5：16行`plot-data.csv`与已提交版本逐值一致（相对误差≤1e-12）。
- 重跑`analyze_e1`：四个配对区间与已提交的analysis JSON完全相同。

没有做跨机器或新venv复现；GPU/驱动不同时的数值一致性未验证。

## 质量检查

对`episodes.csv`核对1520个逻辑键、每键最新attempt、状态、缺失、重复、NaN、成本非负。失败不能从准确率分母消失。至少随机核对5条原始行到summary及图点；正式结果若与开发预检冲突，不改测试集参数。报告主张只对本任务和估算成本口径成立。竞赛提交格式、截止时间与远端地址仍未知，交给接手者后需补齐，但不影响本地实验复现。
