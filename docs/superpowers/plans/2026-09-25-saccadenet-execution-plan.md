# SaccadeNet V0.1 详细执行计划

> **For agentic workers:** 后续获得实施授权后，使用 executing-plans 技能逐项执行；本轮只做计划。步骤使用 `- [ ]` 记录进度。没有明确授权不派发子代理。

**Goal:** 在56小时预算内优先交付可复现的V0-lite、Level 2 E1、三条公平基线、两张核心图、最小回放与可交接报告，并如实说明实验是否支持假设。

**Architecture:** 原图只进入固定金字塔与视网膜；检测、候选反投影、CNN证据、融合B和扫视策略只使用视网膜输出。基线按独立协议访问输入，真值仅供离线监督与评测。统一事件日志贯穿成本、实验、图表和回放。

**Tech Stack:** 计划采用Python 3.11、PyTorch/torchvision、NumPy、SciPy、Pillow、Matplotlib、PyYAML、pytest；OpenCV仅在连通域性能确有需要时加入。安装版本在T01验证Windows/CUDA兼容后锁定，本轮不安装。成本首选显式算子计数并用一个profile工具交叉核对。

**状态：** 仅规划，所有实施步骤未开始。日期2026-09-25。来源：[V0.1](../../saccadenet_v0.1_summary.md)。配套：[进度](../../PROGRESS.md)、[交接](../../HANDOFF.md)、[实验协议](../../EXPERIMENT_PROTOCOL.md)、[决策](../../DECISIONS.md)。

---

## 1. 约束与成功的含义

- 永远不删除文件；需要清理时让用户处理。重跑写新目录，不覆盖或销毁失败证据。
- 用户本轮只要求详细计划。Git与文档已落地；下述代码、命令和任务均为未来实施设计，不代表已实现。
- 硬件已只读确认：RTX 4070 Laptop / 8188 MiB VRAM，约15.7 GiB RAM，驱动580.88，D盘可用约9.7 GiB；CUDA/PyTorch可用性尚未测试。优先本机，不先租GPU。
- 人数暂按单人，H0与实际截止时间尚未确认。用户允许需要更多算力时提出租用建议；真正采购需说明测得瓶颈、预算与时间收益。
- 原方案中的“≥98%”“d′>1”“成本≤4倍”“准确率下降≤3个百分点”“4K–8K崩溃”是待验收目标，不是事实。
- **工程完成**：程序、实验、证据链可复现；**研究成功**：数据支持预注册假设。两者分开记录。没达到G1/G2仍交真实曲线及原因，不能调协议把结果改成成功。

### 交付范围

| 优先级 | 内容 | 启动条件 |
|---|---|---|
| P0 | V0-lite、Level 2 E1、三基线、图4/5、成本明细、最小PNG/GIF回放、报告和复现说明 | 主线，始终优先 |
| P1-A | A/C、协方差诊断、Level 1 E3、oracle基线 | P0四分辨率结果已出且可复现 |
| P1-B | E3b密度与等采样预算两阶段 | P1-A完成且剩余预算充足 |
| P1-C | E2颜色/三档、V0-full、交互回放、E4 | 每项独立评估，不打包承诺 |
| P2 | 视频、端到端微调、目标不存在 | 本次不排实施 |

## 2. 56小时排期与门禁

这是目标时间盒，不是对训练耗时的保证。T01实测资源、T02/S3测吞吐后更新；若剩余时间不足，按降级规则明确降级。

| 相对时间 | 主要工作 | 通过门禁 | 失败处理 |
|---|---|---|---|
| H0–H1 | T01环境、T02范围冻结S5 | 环境可运行、H0和责任明确 | 无GPU先CPU小样本，记录完整E1不可承诺 |
| H1–H6 | T02风险预检：S1/S2/S3/S4；原型可用后并入T03–T08 | 16K采样可行、候选可发现、估出训练/基线耗时 | 停扩展；识别/召回不行先改采样并记录 |
| H6–H14 | T03–T09闭环，T10先打通基线接口 | G1：1080p Level 2闭环，真值隔离与成本记录通过 | H14仍没过：只救闭环，冻结全部P1 |
| H14–H20 | 休息；已验证的训练/标定作业可无人值守 | 保存checkpoint、日志及恢复命令 | 作业失败保留日志，H20处理；无监控不承诺自动恢复 |
| H20–H26 | T10–T13四档冒烟、E1正式批次启动、临时图 | H26四分辨率均有真实结果点，明确样本数 | 缺点则全部时间给E1；临时点不冒充正式100局 |
| H26–H30 | 补齐正式E1、核查区间与成本；启动T14/T15 | G2：三基线齐、正式样本或降级说明齐 | 记录瓶颈、交部分结果，不用Level 1顶替 |
| H30–H38 | 优先T17，再T18；并完成最小回放与报告骨架 | P0结果锁定、有可播放回放 | E3到H36未完成：只做A/B×增益/MAP四组 |
| H38–H43 | 休息；仅运行已验证的剩余批次，可选E4 | 批次持续写日志与checkpoint | 不在睡前启动未测的大矩阵 |
| H43–H46 | T13–T16图、回放、复现演练，冻结代码 | G3：同一Git版本复现小样本与核心图 | 阻断修复有记录；不再新增功能 |
| H46–H54 | 报告、幻灯片、证据对照、离线演示检查 | 内容与实测一致，无未核实强声明 | 缺数据删减论断，不补造数据 |
| H54–H56 | 两次限时排练、校验提交包、人工提交 | G4：包可打开、命令与清单齐、提交回执 | 留最后1小时作上传/格式故障缓冲 |

**租算力触发**：只有T02/T10证明分块和串行仍无法满足内存，或按实测预测P0批次超过H43，才向用户给出GPU型号/显存需求、估计GPU小时、当前报价待确认、数据搬运时间与预计收益。不能把候选算法失败误判为算力不足。

三人配置可分为A（T03–T06数据/感知）、B（T07–T12模型/实验）、C（成本框架、T13–T16报告/演示）；所有人先冻结接口，避免同时编辑同一文件。这只是人员分工建议，不构成派发代理的指令。

## 3. 文件地图与边界

以下路径均相对仓库根目录，当前除文档与Git配置外都尚不存在。只在对应任务开始时创建，不提前堆空文件。

```text
pyproject.toml / requirements-lock.txt       项目入口、锁定依赖与测试配置
configs/base.yaml / e1.yaml / smoke.yaml    所有实验参数，支持配置完整展开
saccadenet/config.py                        配置校验和hash
saccadenet/contracts.py                     数据结构与边界
saccadenet/data/canvas.py                    图像生成与真值输出
saccadenet/data/splits.py                    MNIST索引与种子清单
saccadenet/retina/pyramid.py                 固定平均池化，感知成本
saccadenet/retina/sampler.py                 按环选层、双线性采样与mask
saccadenet/retina/reconstruct.py             只读RetinaOut，生成候选96×96视图
saccadenet/retina/detect.py                  只读RetinaOut，发现/合并候选
saccadenet/models/fovea.py                   11类CNN，可转密集滑窗
saccadenet/models/train_fovea.py             训练、恢复、独立评测
saccadenet/bayes/calibrate.py               μ0/μ1/σ/d′、低辨别力处理
saccadenet/bayes/fusion.py                  P0 B；P1 A/C
saccadenet/bayes/policy.py                  增益、MAP、探索与平局规则
saccadenet/run/episode.py                   Level 2核心循环
saccadenet/run/evaluate.py                  仅这里给预测评分
saccadenet/run/baselines.py                 三条P0基线；P1 oracle
saccadenet/cost/accounting.py               成本事件、同步计时、峰值内存
saccadenet/exp/e1_resolution.py             清单生成、断点恢复、配对汇总
saccadenet/viz/figures.py                    从汇总表生成图4/5
saccadenet/viz/export_replay.py              从trace导出逐帧PNG/GIF
tests/test_*.py                             各任务指定的关键验收测试
reports/spikes/                             S1–S5结论与参数
reports/e1/                                 小型结果表、最终图与复现清单
docs/environment.md / architecture.md       真实运行环境、接口与数据流
docs/references.md / report.md              引文核验、建模报告
docs/REPRODUCE.md                           从环境到图表的命令
```

`runs/`、`data/`、`checkpoints/`、`artifacts/`已规划为忽略目录，存本机大产物。原始文件不删除、不覆盖；权重与结果清单必须能定位到实际文件。

### 核心契约（T01创建，后续保持一致）

| 类型 / 接口 | 内容与约束 |
|---|---|
| `SceneTruth` | 候选/目标位置、标签、digit IDs；只允许生成、离线监督、评测访问 |
| `EpisodeInput` | query、width、height、公开expected_k；不含真值位置/标签 |
| `RetinaOut` | fovea、logpolar、valid masks、sample_xy、fixation、canvas_shape、感知成本；不持有原图/金字塔引用 |
| `Candidate` | stable_id、xy、detector_score、first_seen、last_seen；merge后ID稳定 |
| `Observation` | candidate_id、x或LLR、dprime、有效覆盖率；不可混用Gaussian与Platt语义 |
| `EpisodeLog` | answer_xy可空、reason、steps、候选/后验轨迹、成本、耗时；不携带真值给策略 |
| `Sensor.observe(fixation)` | 传感器内部可持有原图金字塔，向外仅返回RetinaOut |
| `detect_on_retina(ret)` | 返回候选观察值与检测成本，不能额外读取原图 |
| `candidate_view(ret, xy)` | 返回96×96 RGB与valid mask；无效覆盖率>50%则跳过CNN并记录 |
| `FusionB.update(id,x,dprime)` | 质量D=d′²更好才替换；相等质量保留第一条，返回累计LLR |
| `FusionB.update_llr(id,llr,quality)` | Platt退路专用，质量仍取单独标定的d′²；按质量替换已保存LLR，不把LLR再套Gaussian公式 |
| `run_episode(sensor, episode_input, cfg, cal, net)` | Level 2主入口；禁止接收SceneTruth |
| `evaluate(answer_xy, truth)` | 一对一位置匹配/命中/召回评估，真值不回流决策 |

Level 1单独入口接收公开候选坐标，不在Level 2函数中加可选真值后门。训练标签是合法监督，不属于运行时真值泄露。

## 4. 每个任务的执行纪律

每个任务开始在进度看板认领负责人；步骤尽量拆成5–20分钟可验证动作，训练/批量实验作为单独长运行步骤，记录启动时间和恢复命令。功能行为先写有意义的测试再实现；文档无需制造测试。

每个任务结束都必须：执行列出的验收、记录真实输出与产物、勾选已完成步骤、更新PROGRESS/HANDOFF、提交相关文件。提交用明确路径`git add -- <任务文件>`，不使用泛化暂存或清理命令。完成一项不能自动勾选整个阶段。

下述CLI是要在对应任务实现的验收接口；本轮不能执行。命令在仓库根目录、已激活的T01虚拟环境运行。训练示例路径在得到真实产物后写入交接，禁止假定某个权重已存在。

## 5. P0任务清单

### T01 — 资源、环境与契约（约0.75h，依赖：后续开工授权）

文件：`pyproject.toml`、`requirements-lock.txt`、`configs/base.yaml`、`saccadenet/config.py`、`saccadenet/contracts.py`、`tests/test_config.py`、`docs/environment.md`。

- [x] 记录H0/截止时间/参与者；只读检查Python、CUDA、VRAM/RAM、磁盘。保存已确认GPU信息并检测`torch.cuda.is_available()`，不要把驱动支持当作框架可用。
- [x] 安装前估虚拟环境/依赖缓存/数据/权重/日志新增量，要求除此以外至少3 GiB余量；D盘不够则选择另一个可用位置并记录配置，不能自动清理。原始16K不落盘。
- [x] 创建独立`.venv`；按官方兼容信息选择PyTorch/torchvision组合，安装后锁定实测版本，不升级全局环境。因磁盘紧张，实际使用继承系统科学库的venv；偏离详见D16。
- [x] 写配置测试：负K、非法分辨率、τ不在(0,1)、非正T_max应报带字段名的错误；相同配置展开结果hash一致。
- [x] 建立上节契约、基准配置和RGB/坐标约定：xy以左上像素中心为原点、x向右y向下，尺寸顺序统一为W/H，网络张量NCHW float32 [0,1]。
- [x] 运行`python -m pytest tests/test_config.py -q`；预期全通过。运行CPU和GPU各一次固定小张量前向，记录精度/耗时仅作环境验证。
- [x] 提交`chore(T01): define reproducible runtime and contracts`，记录测试与实际硬件，勾选T01。

退出条件：独立环境可运行，契约固定。失败退路：CPU开发可继续，但完整16K资源门禁不能标通过。

### T02 — Day-0风险预检与范围冻结（约5h，依赖T01；使用T03–T08最小原型）

文件：`reports/spikes/S1-policy.md`、`S2-retina.md`、`S3-budget.md`、`S4-pitch.md`、`S5-scope.md`，原型放`scripts/spikes/`并保留。

- [ ] S5（0.25h）：将P0/时间门禁/负责人写入scope，明确本轮主张只针对固定K的语义计算；确认赛事交付要求或记未获得信息。
- [ ] S1（0.75h硬上限）：NumPy生成符合A/B/C各自假设的观测，固定种子比较B增益与MAP，使用预列稀疏/密集两种布局；记录注视、准确率，≥20%优势未出现就降级策略卖点，不反复调密度追结果。
- [ ] S2（2h硬上限）：调用T03/T04最小原型生成一张16K、抽取四角与中心视网膜；在T05检测候选，报告召回/误差；用T06小训练检查数字是否可读，T07粗测d′。不要求2小时完成正式标定，原方案此预算存在风险。
- [ ] S3（1h硬上限）：小图验证滑窗分块，逐级1080p→4K→8K→16K；测单张RAM/VRAM与时间，再估正式批次。单项超时停止扩大规模，保留错误与部分结果；未运行的16K值只能列估计。
- [ ] S4（0.5h）：写成本两层、网络原语、RAM对照、两阶段对照四段说明；用静态草图解释图4/5。可向队友试讲，未获得外部反馈时写“未验证可理解性”。
- [ ] S5复核（0.5h）：把实际资源、不可行点、预计吞吐和必要参数变更写入DECISIONS；所有调参只用开发集。确定继续P0、调整采样或降级部分分辨率。
- [ ] 提交`docs(T02): record feasibility gates and scope`；每个spike必须有数据/命令/结果或明确未通过，不能只写结论。

退出条件：关键风险被测量，不要求全部假设成立。S2/S3是关键路径；S1没有收益不阻塞P0。

### T03 — 可复现画布与数据拆分（约1h，依赖T01）

文件：`saccadenet/data/canvas.py`、`saccadenet/data/splits.py`、`tests/test_canvas.py`、`configs/smoke.yaml`。

- [x] 写测试：同seed逐像素一致；恰1个query；K张卡片在界内且中心距≥288；train/cal/dev/test源数字ID集合不交叉。train/test为独立来源，数值索引允许重叠。
- [x] 实现多尺度1/f近似背景、192卡片、48数字±4抖动，保留K/c/cards_on/contrast参数；16K只逐张生成，uint8为存储格式。
- [x] 真值作为独立`SceneTruth`返回；不把目标颜色/索引编码进预测输入；背景、布局、数字各自使用确定性随机流。
- [x] 设置最大布局尝试次数10,000；不满足约束明确报错。支持低分辨率背景生成+平铺细噪声，并把生成算法版本写清单。
- [x] 运行`python -m pytest tests/test_canvas.py -q`；另运行一次四档生成benchmark，记录16K目标<10秒是否达标及峰值内存，不在CI硬编码机器速度。
- [x] 提交`feat(T03): add reproducible canvas generation`与小型参数/时间报告。

### T04 — 金字塔、抗混叠视网膜与反投影（约2h，依赖T03）

文件：`saccadenet/retina/pyramid.py`、`sampler.py`、`reconstruct.py`、`tests/test_retina.py`、`tests/test_sensor_boundary.py`。

- [x] 写环数测试：修正离散端点后四档79/94/108/122；总采样点19,328/21,248/23,040/24,832。与源表差一环，见D18。
- [x] 构建2×2区域平均金字塔，直到短边<64；奇数边界用分数区域平均。加法次数为解析估计，非精确指令计数，见D19；一次只保留当前画布所需数据。
- [x] 实现r_j=48exp(j·2π/128)，j范围与最外覆盖边界统一；l_j=clip(round(log2(s_j)))。双线性选层采样，角度首尾连续，画布外mask无效。
- [x] 中央凹96×96与外周以共享几何坐标输出；中心/四角/边界注视均可运行。常量图采样保持常量，高频棋盘经过粗层应趋近均值，避免warpPolar假抗混叠。
- [x] 候选视图仅插值RetinaOut样本；中央凹覆盖区域优先用中央凹观测，外周反投影到96×96，空洞保留mask，不读取原图“补清晰度”。
- [x] 验证重建在中心等于已采集中央凹、越界mask正确；无有效覆盖返回全假mask由下游拒绝打分。签名与对象引用测试禁止ret持有原图/金字塔。
- [x] 运行`python -m pytest tests/test_retina.py tests/test_sensor_boundary.py -q`，保存采样/重建对照PNG到spike目录，提交`feat(T04): implement bounded retina observation`。

### T05 — Level 2候选发现与稳定身份（约1h，依赖T04）

文件：`saccadenet/retina/detect.py`、`tests/test_candidates.py`、`reports/spikes/candidate_recall.csv`。

- [x] 写测试覆盖空图、角度接缝单卡、同卡跨帧重复、两张近卡、边缘卡、位置细化；候选ID稳定。重复候选的证据防重计在T08/T09再验收。
- [x] 在视网膜有效样本上做亮度阈值及连通域，把域样本均值通过sample_xy映回画布；中央凹候选并入同一坐标系。原计划的“局部对比度”未启用，开发集目标召回已过门槛，简化记录在D17。
- [x] 按D17的144px阈值去重/细化，禁止读取真实K个中心来补点；检测成本的最终分列在T09验收，此处确认检测函数只接收RetinaOut。
- [x] 在开发集每档20局测多眼候选召回、目标召回、误检和≤20px定位比例；各档目标召回≥95%。首眼结果目前仅在16K少量spike中报告，正式运行继续逐眼记录。
- [x] 固定阈值配置为195/144/5×5；128扇区开发集目标召回过门槛，暂不升级256。E1正式配置冻结在T11。
- [x] 运行`python -m pytest tests/test_candidates.py -q`，提交`feat(T05): discover candidates from retina only`。

### T06 — 中央凹11类CNN与训练恢复（约1h设置+实测训练，依赖T03/T04）

文件：`saccadenet/models/fovea.py`、`train_fovea.py`、`tests/test_fovea.py`、`configs/train.yaml`、`reports/fovea/metrics.csv`。

- [ ] 定义轻量全卷积网络：4级Conv/ReLU/Pool2，通道16/32/64/64，卷积核依次5/3/3/3，全部valid无padding；96输入逐级成为46/22/10/4，再Conv4×4到11类。感受野96、输出步长16，提供patch与dense入口；全图边界统一外部padding，不能在每层padding后声称patch/dense天然等价。
- [ ] 写测试：输出N×11；patch与dense对应窗口输出一致；N=1/多batch均可；固定种子训练恢复能继续同一优化状态。
- [ ] 构建0–9数字加空背景的训练样本；增强来自真实视网膜重建、定位偏差、噪声和部分数字，不能只用理想高清MNIST。
- [ ] 先过拟合小批次检查标签/梯度；再完整训练，记录Adam学习率/批大小/epoch，开发集早停只选开发指标；8GB显存从batch64试起，不足则32/16。
- [ ] 保存模型、优化器、随机状态、epoch和配置hash；用新命名checkpoint保留历史，恢复必须匹配配置。最终分类测试分别报告11类宏平均、总体准确率与空背景误报，清晰数字≥98%单独评估。
- [ ] 运行`python -m pytest tests/test_fovea.py -q`和`python -m saccadenet.models.train_fovea --config configs/train.yaml`；命令完成后将真实checkpoint路径/hash写入报告。
- [ ] 提交`feat(T06): train and validate shared foveal classifier`；权重不进Git，提交指标/清单/恢复命令。

### T07 — 标定、机器视力表与s_min（约1h设置+数据收集，依赖T05/T06）

文件：`saccadenet/bayes/calibrate.py`、`tests/test_calibration.py`、`configs/calibration.yaml`、`reports/calibration/`。

- [ ] 在calibration split采集目标/非目标查询log-odds，初始离心率箱边界为0/24/48/72/96/144/216/324/486/972/17624 px；每箱尽量≥200目标与200非目标，报告实际n及无效mask数量。
- [ ] 拟合共享方差μ0/μ1/σ与d′；实现σ最小值保护、非有限数拒绝、μ1≈μ0或d′<.1跳过，不能除以近零分母。
- [ ] 写合成高斯测试：归一化证据均值应接近0/1、d′估计误差在固定容差内；低信号箱不得生成NaN/无限LLR。
- [ ] 拟合d′(e)曲线并与分箱值画在一起；重点检查e=96处d′>1是否成立。非单调或CNN定位误差大时，使用保守分段单调拟合，记录偏离原因。
- [ ] 在独立标定样本将48像素数字依次缩为4/6/8/10/12/16/24/32/48像素，再按统一管线识别；s_min定义为准确率≥90%的最小测试尺寸，记录离散区间，再登记W_c。
- [ ] Gaussian明显不合时用Platt得到单眼LLR，通过`FusionB.update_llr(id,llr,quality)`保存“质量最高一眼的LLR”，路由d′仍单独标定，不能把LLR误传成x。标明精确高斯推导不再适用。
- [ ] 运行`python -m pytest tests/test_calibration.py -q`、`python -m saccadenet.bayes.calibrate --config configs/calibration.yaml`；提交`feat(T07): calibrate evidence and preregister collapse prediction`。

### T08 — 融合B、路由、探索与停机（约1h，依赖T07）

文件：`saccadenet/bayes/fusion.py`、`policy.py`、`tests/test_fusion.py`、`tests/test_policy.py`。

- [ ] 写B反例测试：低质量支持目标后，高质量否定目标，LLR必须变负；重复同质量不能增加信心；更差质量不能覆盖已有证据。
- [ ] 实现D_i=max d′²、LLR=D_i(x*−.5)，用稳定log-softmax归一化；新候选以中性证据加入，已有候选证据不丢失；位置重置遵循D08。
- [ ] 构造Ω=候选中心+距离<2e_half的两两中点，按画布裁界、去重；G(k)=sum p_i[d′²(e_ik)−D_i]_+，p≤1e−4跳过，确定性平局规则。
- [ ] 实现MAP与D05探索；空候选、单候选、零增益、搜索耗尽都有可解释结果，不能空数组argmax或卡死。
- [ ] 实现D04覆盖门槛、τ与T_max；最后一眼也应生成answer和trace。记录`threshold`/`max_steps`/`search_exhausted`/`no_candidates`等原因。
- [ ] 运行`python -m pytest tests/test_fusion.py tests/test_policy.py -q`；固定玩具可选出两候选中点，若实际标定没有优势则不强造中点演示。
- [ ] 提交`feat(T08): add evidence fusion and guarded search policy`。

### T09 — 闭环、成本事件与真值隔离（约1h，依赖T03–T08）

文件：`saccadenet/run/episode.py`、`evaluate.py`、`saccadenet/cost/accounting.py`、`tests/test_episode.py`、`tests/test_cost.py`、`tests/test_no_truth_leak.py`。

- [ ] 先写故意在读取真值时抛异常的测试对象；Level 2完整循环不应触发；评测可以读取它。禁止Detector、candidate_view、score等接受pyramid/img参数或经全局变量读取。
- [ ] 串联Sensor→检测→反投影→CNN→标定→B→停机/路由，用统一EpisodeLog；生成与评分在循环之外。仅传query，不传目标索引。
- [ ] 实现按组件的成本事件和累计次数；CNN单次profile×实际batch样本数，不遗漏检测、反投影、探索失败眼；不能重复计费金字塔。
- [ ] 写测试：每眼成本非负、累计单调、金字塔仅一次；跳过候选无CNN计费；已知小网络手算与profile一致；失败局也有部分成本。
- [ ] 1080p先用Level 1诊断入口排除融合问题，再切Level 2，至少10局开发数据跑完；这是H14门禁，不要求这10局就达到最终准确率。
- [ ] 运行`python -m pytest tests/test_episode.py tests/test_cost.py tests/test_no_truth_leak.py -q`，保存一局trace；提交`feat(T09): close the measured Level 2 episode loop`。

### T10 — 三条P0基线与16K分块（约2h，依赖T06/T09）

文件：`saccadenet/run/baselines.py`、`tests/test_baselines.py`、`configs/baselines.yaml`、`reports/spikes/baseline_budget.csv`。

- [ ] 用同一个CNN实现全分辨率stride16滑窗，分块建议1024边长起步；halo按实际感受野推导，输出去重，覆盖最后一行/列；不能在全图预先找真值亮卡来减少滑窗量。
- [ ] 写小图整图/分块一致性测试，包含跨块数字与边缘数字；记录halo重复计算，报告实际成本和理想全卷积解析值，不能混成一个值。
- [ ] 实现宽1024的一段式抗混叠降采样，再全卷积定位/分类；输出坐标乘回缩放比，无候选记失败。
- [ ] 实现两阶段：缩图卡片检测→粗查询分数排序→逐个原图96高清crop；校准查询概率越阈值停，否则全部看完选最高；每个读取与判断计费。
- [ ] 两阶段处理空候选、重复检测、越界crop；不允许用真值命中判定来“找到即停”。三基线输出统一EpisodeLog。
- [ ] 运行`python -m pytest tests/test_baselines.py -q`，开发集逐级测四档3局以上。16K内存不够先减tile/batch与串行CPU金字塔；仍失败按租算力/降级门禁执行。
- [ ] 提交`feat(T10): implement all three fair baselines`，补实际吞吐估算和峰值内存。

### T11 — 冻结E1清单与可恢复实验驱动（约1h，依赖T09/T10）

文件：`saccadenet/exp/e1_resolution.py`、`configs/e1.yaml`、`tests/test_experiment_resume.py`、`reports/e1/preregistration.md`。

- [ ] 实现`--dry-run`输出四档×四方法、每条件样本数、总局数1,520、权重/标定hash及预计耗时；不实际启动推理。
- [ ] 实现协议的manifest/config/seeds/episodes/trace/errors产物与唯一键，CSV每局及时flush；重试另记attempt，不覆盖失败。
- [ ] 写中断恢复测试：跑两个种子后中断，再`--resume <run_dir>`不得重复成功行；配置/权重hash改变必须拒绝混跑；截断JSONL末行应报告并在新日志段续写，不删除原文件。
- [ ] 固定原始test digit列表、100个种子与16K全分辨率20种子；登记s_min/W_c、停机阈值、采样/检测参数和成本口径。
- [ ] 运行`python -m pytest tests/test_experiment_resume.py -q`以及`python -m saccadenet.exp.e1_resolution --config configs/e1.yaml --dry-run`；预期无缺失条件、无未指定模型。
- [ ] 完成开发集smoke配置（每档2局×4方法）并检查32局全有记录；失败只修实现或协议并重新冻结，不接触测试集调参。
- [ ] 提交`feat(T11): freeze resumable E1 protocol`；正式运行使用此提交或后续明确修复提交，保存dirty状态，最终结果应来自干净工作树。

### T12 — 正式E1运行与质量检查（约0.5h设置+实测机器时间，依赖T11）

文件：`runs/<run_id>/`、`reports/e1/manifest.json`、`episodes.csv`、`summary.csv`、`docs/PROGRESS.md`。

- [ ] 运行`python -m saccadenet.exp.e1_resolution --config configs/e1.yaml`，先确保所有分辨率/方法都有记录，再扩到目标样本；运行顺序预先写清单。
- [ ] 每阶段检查n、缺失/重复键、NaN、成本单位、目标召回、OOM、阈值与超时分布；故障不从最终分母消失。
- [ ] 若需重启，使用`--resume <真实run_dir>`，把会话/进程、run_dir、命令和已完成数写入HANDOFF。禁止只说“实验在跑”而不留位置。
- [ ] 计算Wilson准确率区间、配对bootstrap差/比值；同时报告完成率，图4成本需能看出失败是否导致少算。
- [ ] 核对G1/G2的点估计、区间和适用条件，给出“支持/不支持/证据不足”；16K基线20局在图中标n=20。
- [ ] 提交`data(T12): record measured E1 results and provenance`，仅加入小型摘要/清单/必要逐局CSV；原始大trace与权重不进Git。

### T13 — 图4/图5和数据自检（约1h，依赖T12；可先用开发结果验证画图）

文件：`saccadenet/viz/figures.py`、`tests/test_result_schema.py`、`reports/e1/fig4_cost.png`、`fig5_accuracy.png`、`cost_breakdown.csv`。

- [ ] 图表只从summary/episodes读值，不在绘图脚本手填结果；schema要求方法、分辨率、n、估计类型、均值/区间、单位齐全。
- [ ] 图4双对数画面宽度vs语义FLOPs，附感知/其他ops/总浮点运算/延迟表；解析外推使用不同线型，不能冒充实测点。
- [ ] 图5准确率与95%区间，随机猜测1/12仅在固定K任务下作为参考；失败计入准确率。缺条件显式标缺，不能插值补成数据。
- [ ] 按相同seed核查至少5条原始episode→summary→图点；人工查看标签、图例、字体、裁切、单位、样本数和来源脚注。
- [ ] 运行`python -m pytest tests/test_result_schema.py -q`和`python -m saccadenet.viz.figures --run <真实run_dir> --out reports/e1`；图4/5必须同时含四方法和四档，降级例外明确列出。
- [ ] 提交`viz(T13): render source-backed resolution results`；开发图如已生成保留原位置，以不同文件名区分最终图。

### T14 — 最小离线回放（约1h，依赖T09/T12，H46前完成）

文件：`saccadenet/viz/export_replay.py`、`reports/replay/manifest.json`、`artifacts/replay/<episode_id>/`。

- [ ] 选一局有代表性的成功例与一局失败例，记录选择理由，不声称是随机抽样；从已保存trace生成，不重新运行另一版本模型。
- [ ] 主画面展示缩略图、编号注视轨迹、已发现候选与后验、累计semantic/sensing计数；最终答案与真值由评测层叠加，不能回流推理。
- [ ] 每步可附中央凹与logpolar观察，表明实际网络输入；P0不做交互播放器，不把高清完整图嵌入每帧。
- [ ] 运行`python -m saccadenet.viz.export_replay --run <真实run_dir> --episode <实际ID> --format png`，逐帧PNG为底线，GIF可选；断网验证至少一个完整回放能打开。
- [ ] 检查帧数与trace步数一致、坐标/数字标注一致、成本终值与CSV一致；提交`demo(T14): export auditable offline replay`及清单，小型预览可进Git。

### T15 — 数学说明、文献核验与报告（约2h骨架，H46后集中完善，依赖T02；结果部分依赖T12/T13）

文件：`docs/architecture.md`、`docs/references.md`、`docs/report.md`、`reports/presentation-outline.md`。

- [ ] 先写问题/假设/采样推导/观测模型/融合B/成本口径/实验方法骨架；所有图表位置指向真实产物，不写预定成功数字。
- [ ] 核对N(R)、离散环数、边界mask、d′与LLR、B按质量替换、ELM增益非Shannon熵减，以及固定K/增长K/无候选三种条件。
- [ ] 逐条找一手论文/官方来源核验RAM、Najemnik-Geisler、采样定律、MNIST等关键引用；原方案引文列表未核实项不进入已验证引用，创新性措辞保持可支持范围。
- [ ] 补真实实验配置、split、样本数、区间、硬件、时间、候选召回和成本分解；明确Level 2未知目标质量未建模、D04保护的局限、确定性重复观测问题。
- [ ] 对照三条基线解释图4/5，即使两阶段更好也如实写；“总代价不随分辨率增长”“首次logR”“静态模型C一定精确”等表述不得出现。
- [ ] 写四个答辩问题的短答：前端是否读全图、网络创新在哪里、与RAM区别、与两阶段区别；每个答复链接到证据或文献。
- [ ] 按赛事实际页数/格式完成报告、幻灯片提纲；人工核查图号/数值/来源/引用一致，提交`docs(T15): explain measured findings and limitations`。

### T16 — 复现、冻结、交接与提交包（约2h，依赖T13–T15）

文件：`docs/REPRODUCE.md`、`docs/HANDOFF.md`、`reports/release-manifest.json`、最终报告/幻灯片/回放索引。

- [ ] H43前记录最终依赖、权重hash、标定hash、数据来源、命令和产物路径，检查大文件实际可访问；离线演示不依赖云服务。
- [ ] 从新的独立工作目录或环境做smoke复现（不删除旧目录），运行完整测试与32局冒烟；再用保存的正式CSV重画核心图，数据数值应一致。
- [ ] 对照P0覆盖表逐项核查；未达工程或研究门槛均写明，不能为提交包好看而勾选未完成项。
- [ ] H46冻结代码，创建`v0.1-demo`本地标签（若已存在则使用新版本号，不覆盖标签）；之后阻断修复必须新提交并重跑受影响验收。
- [ ] 更新HANDOFF：最后提交/分支、真实运行命令、未提交改动、进程状态、未解决问题、下一步和产物访问方法；核查`git status --short --branch`。
- [ ] 校验报告/PDF或赛事格式、图、回放可打开；核对提交包checksum和目录，不打包凭据或完整训练缓存。两次限时排练。
- [ ] 用户/队员按赛事流程提交，保存回执；未真正提交不得勾选“赛事已提交”。提交`docs(T16): finalize reproducibility and delivery handoff`。

## 6. P1任务（不占用P0救火时间）

### T17 — E3融合与策略诊断（预算4h，依赖完整P0）

文件：`saccadenet/bayes/fusion.py`、`saccadenet/exp/e3_fusion.py`、`tests/test_fusion_ac.py`、`configs/e3.yaml`、`reports/e3/`。

- [ ] 实现A累计LLR；C维护W/S与σ_u，限制σ_u²<1/d′²(0)，W=0时精度定义0，σ_n²≤0报告拟合无效而非静默修正。
- [ ] 测试A为C的σ_u→0极限、C饱和上限、重复确定性观测不能视为独立残差证据；标明C的适用范围。
- [ ] 同候选同类别跨离心率协方差诊断，比较0/较清楚观察方差/常数三个模型；采集与拟合只用校准集，最终校准评价用留出集。
- [ ] Level 1配对跑A/B/C×增益/MAP/随机/栅格与oracle_crops；扫τ={.5,.7,.8,.9,.95,.99}，每条件100局，测时间后再决定可完成子矩阵。
- [ ] 运行`python -m pytest tests/test_fusion_ac.py -q`；E3命令固定`python -m saccadenet.exp.e3_fusion --config configs/e3.yaml`；报告Brier、可靠性、准确率、平均眼数与配对区间。
- [ ] H36尚未完成则锁定A/B×增益/MAP四组、τ=.95，标为缩减E3；提交`exp(T17): compare fusion assumptions and search policies`。

### T18 — E3b密度与等预算两阶段（预算2h，依赖T17）

文件：`saccadenet/exp/e3b_density.py`、`configs/e3b.yaml`、`reports/e3b/`。

- [ ] 在16K构造最小间距288/576/1152三组布局，K=12、同数字和背景清单，各100局；记录实际最近邻距离，不让不可放置布局悄悄改变K。
- [ ] 比较SaccadeNet、1024两阶段及等预算均匀粗图；等预算按实际retina采样点求宽高，210×118只是约数，报告真实点数。
- [ ] 统一高清确认规则、计数与校准评估范围；比较两阶段初次粗看预算与逐眼预算时说明时间维度，不能混为总预算相同。
- [ ] 运行`python -m saccadenet.exp.e3b_density --config configs/e3b.yaml`，报告密度下召回、高清次数、成本、准确率与区间；提交`exp(T18): measure density tradeoffs against coarse-to-fine`。

### T19 — 其余扩展的独立入口（每项限时，依赖T17/T18且H43前）

- [ ] E2（上限2h）：先只做卡片开、16K K={4,8,16,32,64}；卡片关闭在1080p/4K以64格扫，记录T_max与失败。未跑到更大图只能列外推。
- [ ] 颜色通道（上限1h）：用retina采样颜色单独标定，扫c={0,.25,.5,1}；验证与数字证据独立性后才相加，不足则标经验模型。
- [ ] V0-full（上限2h预检）：`models/periphery.py`吃logpolar，角度环形padding，发现+查询两头；与lite共用协议。d′或召回不优于lite立即结束预检，记录负结果，不替换P0。
- [ ] 交互回放（上限1h）：只消费已导出的JSON/图片，增加播放进度与网络视角开关；不接实时推理。“你来当眼睛”仅有额外时间再做。
- [ ] E4（上限1h配置）：中央凹64/96/128、扇区64/128/256、数字32/48/64与背景对比度做单因素扫，受影响权重/标定须重建，不复用失配标定。没时间就不启动。
- [ ] 每项分别记录文件、配置、种子、命令、结果与提交；没做的保持未勾选，不把T19整体当成单一完成项。

## 7. 覆盖、依赖与验收索引

主依赖：T01→T03→T04→T05/T06→T07→T08→T09→T10→T11→T12→T13；T14依赖trace、T15可先写方法，T16收口。T02是跨模块限时可行性原型门禁，不要求正式模块先全部完成；原型验收后重用或整理，保留原文件。

| V0.1要求 | 对应任务 | 可核查证据 |
|---|---|---|
| §2画布与§14生成器 | T03 | 几何/split/复现测试，生成时间 |
| §4/§5.1采样与反投影 | T04 | 环数、抗混叠、mask、输入隔离测试 |
| §14 Level 2候选检测 | T05/T09 | 召回/定位报告，真值访问陷阱 |
| CNN、标定、机器视力表 | T06/T07 | 留出准确率、d′分箱/拟合、checkpoint清单 |
| B、增益、MAP、停机 | T08/T09 | 证据反例、空候选/单候选/超时测试 |
| §6成本与三基线 | T09/T10 | 手算/profile核对、分块等价、成本事件 |
| §8 E1与图4/5 | T11–T13 | 冻结协议、1,520局计划、真实CSV、区间与图 |
| §10最小回放 | T14 | trace一致性和离线打开检查 |
| 数学、引用与提交 | T15/T16 | 文献核验表、报告、复现记录、交付清单 |
| §11 S1–S5与砍量 | T02/时间表 | spike报告、门禁决策与失败证据 |
| §14 P1 | T17–T19 | 各自实验配置与报告，未做保持空框 |
| §14 P2 | 不实施 | 明确延期，不混入P0任务 |

## 8. 最低交付与降级口径

**完整P0**：四分辨率、四方法，达到约定样本规模；代码/数据/成本/图/回放/报告/复现链齐。科学假设不成立时照样呈现真实结果，但研究目标栏标未达或证据不足。

**部分交付**：若16K或全分辨率基线未完成，交实际1080p–8K数据、失败原因、采样推导、spike与视力表；图中缺失明确标注，名称写“部分扩展实验”，不标完整P0。

**最后退路**：如果Level 2检测不可靠，只能交Level 1融合诊断与理论原型；撤回图4/5对应的可扩展视觉搜索论断，明确主目标未完成。

不因负结果删文件、不为达到截止时间伪造结果、不以没有数据的演示替代实验。这些约束贯穿每次接手。
