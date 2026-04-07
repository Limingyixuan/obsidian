有关transformer的问题：
网址链接1：
[(4 封私信 / 42 条消息) Transformer面试题总结101道题 - 知乎](https://zhuanlan.zhihu.com/p/438625445)

## visiontransformer
1、图片变为patch
2、添加位置编码，位置编码为1D可学习的位置编码
3、添加类别编码
4、前向传播
5、


### KL散度和交叉熵：
![](image/Pasted%20image%2020260407161529.png)

![](image/Pasted%20image%2020260407161536.png)

![](image/Pasted%20image%2020260407161545.png)


![](image/Pasted%20image%2020260407161655.png)

KL散度是衡量标准分布和模型预测分布之间的差距的值，
KL散度在分类任务中，化简后得到的公式为交叉熵减去真实分布的熵，但因为真实分布的值是一个固定值所以之和交叉熵有关


2、大模型的幻觉问题的成因和解决方法

模型

### 分词的方式：
BPE
wordpiece
sentencepiece

### batch N和LN的区别