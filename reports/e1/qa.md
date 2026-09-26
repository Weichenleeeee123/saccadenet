# E1 正式运行质量检查（T12/T13）

检查日期：2026-09-25（Asia/Taipei）。检查人：Claude（接手 Codex）。本文件只记录核对证据，不重新解释结论；结论见 [report](../../docs/report.md)。

## 被检对象

- 运行目录：`runs/20260925T152349319841Z-a6a00495-e1-final`（Git 忽略，本机保存）；manifest 记录 `git_head=a6a0049`、`git_dirty=false`、`split=final_test`、权重 SHA256 `3d80b18c…`。
- 审计与配对区间：[`analysis-…153843082396Z.json`](analysis-20260925T152349319841Z-a6a00495-e1-final-20260925T153843082396Z.json)，由 `python -m saccadenet.exp.analyze_e1 --run <run_dir>` 生成。更早的 `analysis-…153655070563Z.json` 内容相同，是同一次审计的首次输出，保留备查。
- 正式图目录：[`…-20260925T153721798894Z/`](20260925T152349319841Z-a6a00495-e1-final-20260925T153721798894Z/)。`…-20260925T153655205949Z/` 是第一次出图失败的残留：Wilson 下界的浮点舍入产生了负误差棒，只写出了图 4，图 5 没有生成。修复方法是对误差棒做非负裁剪（见 `saccadenet/viz/figures.py`）。残留目录只供审计，不用于报告。

## 账本审计（analyze_e1）

计划键 1,520；逐局行 1,520；最新键 1,520；画布 seed 行 400；trace 1,520。缺失、意外键、失败的最新尝试、重复成功、非有限或负数成本、seed 缺漏或多余、trace 成本不一致、步数不一致、成本回退，这些检查项全部为空。`errors.jsonl` 大小为 0。

## 逐局 → summary → 图数据 → trace 抽核

命令：

```powershell
.venv\Scripts\python.exe -m scripts.qa_e1_crosscheck --run runs/20260925T152349319841Z-a6a00495-e1-final --plot reports/e1/20260925T152349319841Z-a6a00495-e1-final-20260925T153721798894Z
```

输出（2026-09-25 实跑，`ALL_CONSISTENT=True`）：

| 方法 | 宽 | n | 命中(逐局/summary/图) | 语义FLOPs均值(逐局/summary/图) | 一致 |
|---|---:|---:|---|---|---|
| saccadenet_lite | 1920 | 100 | 98/98/98 | 4.33262e+08/4.33262e+08/4.33262e+08 | 是 |
| saccadenet_lite | 15360 | 100 | 100/100/100 | 4.43126e+08/4.43126e+08/4.43126e+08 | 是 |
| full_res_sliding | 15360 | 20 | 19/19/19 | 1.28328e+12/1.28328e+12/1.28328e+12 | 是 |
| downsample_1stage | 3840 | 100 | 0/0/0 | 5.00452e+09/5.00452e+09/5.00452e+09 | 是 |
| two_stage | 15360 | 100 | 100/100/100 | 1.08661e+09/1.08661e+09/1.08661e+09 | 是 |

| 逐局键 | steps(CSV/trace) | 语义FLOPs(CSV/trace末步) | 感知字节(CSV/trace) | 一致 |
|---|---|---|---|---|
| saccadenet_lite:15360:30000 | 30/30 | 465157619/465157619 | 533074680/533074680 | 是 |
| saccadenet_lite:15360:30001 | 36/36 | 697065744/697065744 | 533521656/533521656 | 是 |
| saccadenet_lite:15360:30002 | 28/28 | 638604144/638604144 | 532925688/532925688 | 是 |
| saccadenet_lite:15360:30003 | 33/33 | 581111952/581111952 | 533298168/533298168 | 是 |
| saccadenet_lite:15360:30004 | 32/32 | 638900118/638900118 | 533223672/533223672 | 是 |

另外，用独立脚本按 `episodes.csv` 复算了 16 组（4 方法 × 4 档）的命中率、语义/感知 FLOPs、读取字节和延迟，与 [report](../../docs/report.md) 表格逐项一致。

## 分布检查

SaccadeNet 各档终止原因与未命中局：

- 1920：threshold 99，search_exhausted 1；目标候选 48px 召回 100/100；未命中 30002（threshold，2 眼）、30061（threshold，6 眼）
- 3840：threshold 94，search_exhausted 6；召回 100/100；未命中 30021（threshold，12 眼）、30031（6 眼）、30045（7 眼）、30077（9 眼），均为 threshold
- 7680：threshold 95，search_exhausted 5；召回 100/100；未命中 30029（threshold，23 眼）、30061（search_exhausted，37 眼）、30090（threshold，32 眼）、30095（search_exhausted，38 眼）
- 15360：threshold 98，search_exhausted 2；召回 100/100；无未命中

**发现：** 10 个未命中里，8 个是后验越过 τ=0.95 后停在错卡上。也就是说，它们是过度自信的错停，不是没找到目标，也不是超时。标称 95% 的停机阈值在这些局里没有兑现。每档只有 100 局，不足以估计校准误差，但报告里不能把 τ 说成"95% 置信"。

## 已知字段限制

- `peak_rss_bytes` 是每局结束时读取的当前 RSS，不是过程峰值。
- 全分辨率滑窗的 `sensing_flops=0` 只表示没有换算读图运算，它实际读取了全图（见 `sensing_bytes`）。
- OpenCV 缩放、连通域等 FLOPs 是解析近似（D19/D22）。
