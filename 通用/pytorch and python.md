### 广播机制
如果遵守以下规则，则两个tensor是“可广播的”：
每个tensor至少有一个维度；
遍历tensor所有维度时，从末尾随开始遍历，两个tensor存在下列情况：
tensor维度相等。
tensor维度不等且其中一个维度为1。
tensor维度不等且其中一个维度不存在。
如果两个tensor是“可广播的”，则计算过程遵循下列规则：
如果两个tensor的维度不同，则在维度较小的tensor的前面增加维度，使它们维度相等。
对于每个维度，计算结果的维度值取两个tensor中较大的那个值。
两个tensor扩展维度的过程是将数值进行复制。

原文链接：https://blog.csdn.net/m0_52650517/article/details/119913625、

#### 数组
#### 1、randint方法：
Python **random.randint()** 方法返回指定范围内的整数。
语法：
random.randint(start, stop)
#### 2、torch.randint :
生成指定大小的矩阵，且矩阵的值在指定的范围内随机。：
torch.randint(0,2,(3,2))：
生成一个3 * 2的矩阵且，矩阵内的值从0到2之间随机，不包括2。