端到端无人机小目标检测模型，使用了DETR模块，目前的DETR类的模型大多适用于natural images，这种模型并不适用于UAV images detection
UAV目标检测有两个难点，Small和Occlusion（闭塞）。
因此，对于UAV目标检测，detailed feature extraction 十分有效。目标的特征未必足够多，所以可以考虑结合目标和周围环境的信息（==在SPAR模型中也提到这一点==）。
模型提出了一种以