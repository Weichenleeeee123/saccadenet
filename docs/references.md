# 文献核验记录

核验日期：2026-09-25。只把本轮直接使用的一手论文及数据集主页列为已核实。原V0.1附录C自述“凭记忆整理”，其余条目未自动继承为已核实引用。

| 来源 | 已核实元数据与本项目可支持的用途 | 不能据此宣称 |
|---|---|---|
| [Najemnik & Geisler, *Nature* 2005](https://www.nature.com/articles/nature03390) | *Optimal eye movement strategies in visual search*, 434:387–391，DOI `10.1038/nature03390`；理想搜索者、视野可检测性与注视选择的理论动机。 | 我们的确定性静态CNN观测必然满足其噪声假设；我们的策略已达生物最优。 |
| [Najemnik & Geisler, *Vision Research* 2009](https://pubmed.ncbi.nlm.nih.gov/19138697/) | *Simple summation rule for optimal fixation selection in visual search*, 49(10):1286–1294，DOI `10.1016/j.visres.2008.12.005`；ELM以检测力平方加权的简化规则。原文依赖每眼噪声假设。 | 本项目的增益函数就是精确Shannon熵减；静态重复观察提供独立新证据。 |
| [Mnih et al., NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/3e456b31302cf8210edd4029292a40ad-Abstract.html) / [论文PDF](https://proceedings.neurips.cc/paper/2014/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf) | *Recurrent Models of Visual Attention*；多分辨率glimpse与局部高分辨率处理是直接前例。论文说明计算可相对输入尺寸受控。本项目只比较采样律、固定金字塔成本拆分和此处的合成E1协议。 | “首次只看局部”或“首次使语义计算与图幅脱钩”。从逐级增大的patch推论其尺度数随覆盖范围对数增长是我们的推断，不是论文直接报告的复杂度定理。 |
| [LeCun、Cortes、Burges MNIST主页](https://yann.lecun.org/exdb/mnist/index.html) | 官方主页列60,000训练图与10,000测试图，28×28尺寸；本项目train源中再划分train/calibration/development，官方test留给最终评测。 | 画布由MNIST构成就等价于自然图像验证。 |

原附录C其余条目（Koch、Horton/Hoyt、Wald、Blackwell、Treisman/Gelade、Graves、Jaderberg、Jang、Dosovitskiy、Wu/Xie、2026年Cell连接组）尚未逐条核验，当前报告不引用其具体断言。特别是新闻/连接组条目不用于宣称模型的生物学真实性或创新性。
