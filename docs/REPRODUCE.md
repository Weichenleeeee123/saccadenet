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

## 质量检查

对`episodes.csv`核对1520个逻辑键、每键最新attempt、状态、缺失、重复、NaN、成本非负。失败不能从准确率分母消失。至少随机核对5条原始行到summary及图点；正式结果若与开发预检冲突，不改测试集参数。报告主张只对本任务和估算成本口径成立。竞赛提交格式、截止时间与远端地址仍未知，交给接手者后需补齐，但不影响本地实验复现。
