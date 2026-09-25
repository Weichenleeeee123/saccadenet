# 实施环境（T01）

记录时间：2026-09-25 22:13–22:27 Asia/Taipei。H0按开始实现时22:13计；用户尚未给具体比赛截止时间与队员人数，执行暂按单人和56小时相对时间。

| 项目 | 实测 |
|---|---|
| 系统 | Windows / PowerShell；Python 3.12（系统另有3.13） |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB显存，驱动580.88 |
| RAM | 约15.7 GiB物理内存 |
| 磁盘 | 开工前D盘9.7 GiB、C盘1.8 GiB；GPU依赖安装后D盘5.34 GiB |
| 虚拟环境 | `.venv`，Python 3.12，启用`--system-site-packages`以复用已安装科学库；该环境并非完全隔离，必须按锁文件复核版本 |
| PyTorch | 虚拟环境2.6.0+cu126；CUDA可用，64×64 GPU全1张量求和4096.0 |
| torchvision | 0.21.0+cu126，可正常导入 |
| 其他依赖 | `requirements-lock.txt`给出本机已验证版本 |

预检问题：系统Python 3.12原有CPU版PyTorch 2.5.0，搭配CUDA版torchvision 0.21.0，`torchvision::nms`缺失。虚拟环境只覆盖PyTorch为配套GPU版，未改全局安装。版本组合依据[PyTorch官方历史版本安装表](https://docs.pytorch.org/get-started/previous-versions/)。GPU wheel约2.32 GiB；安装时临时目录设到工作区`.tmp`以避开空间很少的C盘，pip结束后剩余空间已检查。

复核命令（在工作区根目录运行）：

```powershell
.\.venv\Scripts\python.exe -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available()); print(torch.ones((64,64),device='cuda').sum().item())"
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
```

计划的完整实验可能需要1 GiB以上额外中间/运行文件。新增数据、权重与运行日志前继续检查D盘余量；不自动清理现有文件。16K完整画布仅在内存生成。
