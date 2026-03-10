
## 一、yolov2：
1、引进了batch nomalization

2、使用了预训练，先用224 * 224的图片进行训练160，再换成448 448的训练10输入图片的尺寸修改

3、引进了改进的anchor box-通过训练集聚类获得先验的anchor box的边框大小：
**在每个grid cell预先设定一组不同大小和宽高比的边框，来覆盖整个图像的不同位置和多种尺度**。

## 二、yolov3
保留的v1和v2的特点：
1、采用 leaky ReLU作为激活函数

v3的改进之处：
- 多尺度预测（引入FPN）多尺度的特征图
- 更好的backbone(darknet-53,类似于[ResNet](https://zhida.zhihu.com/search?content_id=213655736&content_type=Article&match_order=1&q=ResNet&zhida_source=entity)引入残差结构)**没有池化层和全连接层**，张量的尺寸变换是通过改变卷积核的步长来实现的
- 分类器不再使用softmax(darknet-19中使用)而是使用logistic分类器，损失函数中采用binary cross-entropy loss（二分类交叉损失熵）以适用重叠类别物体的检测

## 三、yolov4
1、mosaic数据增强：丰富数据集，有利于小目标的检测性能
2、 CSPDarknet53
3、mish激活函数
4、FPN加PAN：在加了一层下采样加拼接，进一步提升特征提取的能力

## 四、yolov5
1、继承mosaic数据增强
2、自适应锚框计算：调整size，k-means聚类，遗传算法
3、backbone的focus结构：
对图片进行切片操作（一张图片每隔一个像素拿一个值）得到相当于12通道的图片（原本是三通道）
信息不丢失的情况下提升计算力
neck的网络也采用了FPN+PAN的结构，不同是yolov5的neckcspnet用的是csp2结构，加强网络的特征融合能力。
4、非极大值抑制
5、损失函数为cIOU，及考虑框与框的中心距离，也考虑了框与框的长宽的比例
SPPF：
spp：对任意输入大小的图片进行处理，提高网络泛化和效率

五、yolov7：
1、模型重参化
用不同数据训练的多个相同模型，对这几个模型进行权重平均
不同迭代次数的模型进行权重加权平均
2、模块重参化

3、模型缩放：
- input size（输入图像大小）
- depth（层数）
- width（通道数）
- stage（特征金字塔数量）

## YOLOv8
backbone由yolov5的c3模块改成c2f模块：
c2f模块多了tensor的split拆分，其中一部分经过n层的bottleneck的
**原始拆分的两个分支** + **n 个 Bottleneck 的输出**全部`Concat`，此时通道数为`0.5×(n+2)×c_out`（包含了更多细节和梯度信息）。
