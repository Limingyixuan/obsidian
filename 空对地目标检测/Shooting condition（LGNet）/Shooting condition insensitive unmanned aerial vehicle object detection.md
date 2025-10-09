



![](Pasted%20image%2020250908092204.png)




![](Pasted%20image%2020250908092218.png)


模型架构：
基础模型为Faster R-CNN + FPN，作为目标检测的模型

在训练过程中加入了LGNet语言指导的训练部分：1、文本提示嵌入微调，2、对齐策略


1、文本提示词嵌入：
先设置好固定的文本模板：
An {alt-condition} altitude {view-condition} view of a {weather-condition} day taken by a drone

在模板中有三个词是需要进行替换的：使用CLIP微调，具体如下

首先三个词分别为：角度，高度和天气，三个中情况的分类分别是：晴天、夜晚、雾}、{高、中、低} 和 {鸟、侧、前、前，所以生成3 * 3 * 4 = 36个情况，即36个编码向量，后续输入到CLIP模型中。

随后，使用CLIP模型对这36个编码向量进行微调，冻结其他部分，只留下PromptLearner这一部分进行微调，用所有的图片数据集与36个编码进行匹配，每个图片得到36个对应得分，

微调的具体理解：[Text prompt embedding fine-tuning](Text%20prompt%20embedding%20fine-tuning.md)