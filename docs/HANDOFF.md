# 接手说明

更新时间：2026-09-25（Asia/Taipei）。本次交接点：T01、T03–T05、T10–T11已完成；T06–T09部分验收待补，准备从干净提交运行E1。

## 现在实际有什么

- 主仓库：`D:\saccadenet`；实施工作树：`D:\saccadenet\.worktrees\implementation`，分支`feat/implementation`。从工作树继续开发，主工作区`main`保留规划。
- 原始文档：`docs/saccadenet_v0_summary.md`、`docs/saccadenet_v0.1_summary.md`，原样保留；快照提交 `5322323`。
- 新增规划文档见根目录 README。规划提交可通过 `git log -5 --oneline` 查看，标题为 `docs: add execution plan and handoff tracking`。
- 已有T01配置/契约、T03–T05代码和开发召回报告。T06–T09模型、标定、闭环在提交`d222f1d`中；T10三基线及T11可续跑实验驱动正在验收。MNIST已下载到忽略目录`data/mnist`。**尚无正式E1结果，也没有已验证的科研结论。**
- v1主权重：`checkpoints/fovea/20260925T145236155331Z-b107b36e/epoch-008.pt`，SHA256 `3d80b18c01e0cb25fb3cc01dea891a5a5f1dcb35bbd8676102e490718bf8eda9`；retina标定：`reports/calibration/calibration-20260925T145532233348Z-3d80b18c-fit.json`；两阶段Platt：`reports/calibration/two-stage-platt-20260925T151159460820Z-3d80b18c.json`。权重被Git忽略，跨机交接必须实际复制。
- 开发32局冒烟首次全部因trace序列化报错，错误保留于`runs/20260925T151457474939Z-d222f1d5-e1-smoke`；修复后同目录第二次attempt 32/32成功。随后新增`sensing_bytes`字段，旧目录schema不可续跑。最终schema另建`runs/20260925T151923188489Z-d222f1d5-e1-smoke`，32/32成功、`summary-2.csv`有理想滑窗成本列。
- 没有远端；不能认为项目已备份到 GitHub。
- 已核验：RTX 4070 Laptop / 8188 MiB VRAM，驱动580.88，RAM约15.7 GiB；当前D盘剩余约5.22 GiB。项目venv中PyTorch 2.6.0+cu126已通过GPU小张量测试。16K全分辨率滑窗开发首局推理约6.8秒，画布生成约4.6秒，尚无需租算力。

## 接手后的第一步

```powershell
Set-Location -LiteralPath 'D:\saccadenet'
git status --short --branch
git log -5 --oneline
git remote -v
```

随后进入实施工作树，阅读最新`AGENTS.md`、`docs/PROGRESS.md`、详细计划与实验协议。用户已明确授权持续实施。

H0记录为2026-09-25 22:13 +08；实际截止时间和队员仍未知，按单人56小时相对排期。当前从干净提交运行正式1,520局E1。默认V0-lite、融合B、Level 2 E1。

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

当前下一项：在实施工作树运行`.venv\Scripts\python.exe -m saccadenet.exp.e1_resolution --config configs/e1.yaml`，记录run_dir与会话，再做T12质量检查。T02的S1玩具策略与S4答辩草稿尚未完成；T06分类完整指标及真实视网膜重建增强、T07曲线与分布诊断尚未完成，不能勾选。查看`git status --short --branch`确认未提交改动；不要把`runs/`已忽略误认成丢失。
