
## GPT1：
自回归和非自回归：
![](LLM/image/Pasted%20image%2020251117151932.png)

GPT是自回归模型


We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task
从无标记的文本中提取信息有巨大潜力

从无标记的文本中提取信息面临的挑战：
1、优化目标的不确定
2、如何将学习到的文本表示如何迁移到具体的下游任务还未达成共识

GPT的贡献：
- **「预训练 + 微调」范式的验证（基于 Transformer 解码器）**

- 采用了**仅**由 Transformer 解码器堆叠的架构（使用 Masked self-attention 从左到右预测下一个词），在大规模未标注语料上进行**生成式预训练**。
- 随后，模型在下游任务（文本蕴含、文本分类、问答等）上通过有监督**微调**来适配不同场景，最终在 9/12 的任务上取得了 SOTA，证明了 **Transformer 架构**在语言建模上的可行性。
- 虽然在 GPT 出现之前已有基于**预训练**词向量（Word2Vec [[MCCD13]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1301.3781)、GloVe [[PSM14]](https://link.zhihu.com/?target=https%3A//nlp.stanford.edu/pubs/glove.pdf)）或 ELMo 等双向语言模型的类似思路，但 GPT-1 **首次**在一个**大规模、纯 Transformer 解码器**上系统性地验证了「预训练 + 微调」范式的有效性，为后续基于 Transformer 架构的预训练语言模型（如 BERT、T5）奠定了基础。

- **引入统一的任务输入格式**

- 通过在输入文本中添加特殊标记以及拼接文本，将不同下游任务（文本蕴含、问答、情感分析等）的结构化输入统一转换为**连续序列**的形式。
- 这种方法减少了为不同任务**单独设计模型结构**的需求，仅通过调整输入格式即可适应不同任务，使得同一个预训练语言模型可以在不同任务之间复用。

![](image/Pasted%20image%2020260330152106.png)


![](image/Pasted%20image%2020260330151330.png)


## BERT：
12个或者24个transformerencoder堆叠而成

Embedding：
三种
1、Token Embeddings 是词向量，第一个单词是CLS标志，可以用于之后的分类任务
2、Segment Embeddings 用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务：
Bert 能够处理句子对的分类任务，这类任务就是判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入模型中，Bert 如何区分一个句子对是两个句子呢？答案就是 Segment Embeddings。前一个句子是将0赋值给每个token，后一个句子是将1赋值给每个句子

