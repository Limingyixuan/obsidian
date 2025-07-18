端到端无人机小目标检测模型，使用了DETR模块，目前的DETR类的模型大多适用于natural images，这种模型并不适用于UAV images detection
因此，对于UAV目标检测，detailed feature extraction 十分有效。目标的特征未必足够多，所以可以考虑结合目标和周围环境的信息（==在SPAR模型中也提到这一点==）。
贡献：
模型提出了一种以频域为中心的下采样策略。

### 相关工作：
1、UAV目标检测：
UAV目标检测有两个难点，Small和Occlusion（闭塞）。
过去的UAV-OD关注于提取空间域的特征和上下文的信息，而频域的特征没有得到充分利用
以往的UAV-OD模型用yolo系列的模型进行检测，这些模型需要用NMS进行后续处理，而NMS会降低推理速度，影响推理精度。
[NMS](NMS.md)笔记链接
而RT-DETR是第一个实时的端到端的目标检测模型，且消除了NMS的影响。


