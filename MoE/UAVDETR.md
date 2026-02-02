
## 初始模型：
#### 训练1：291epoch：
![](image/Pasted%20image%2020260121094325.png)

### 400epoch：
runs_origin/train/exp
#### 测试集：
AP50:0.421
mAP:0.25
![](image/Pasted%20image%2020260123093546.png)


#### 验证集：
AP50：52.5
mAP：32.9
![](image/Pasted%20image%2020260123140902.png)



## 加上初始moe：
runs_moe1/train/exp5
#### 测试集
AP50:0.25
mAP:0.422
![](image/Pasted%20image%2020260122094245.png)

#### 验证集
AP50：53.3
mAP：33.5
![](image/Pasted%20image%2020260123141142.png)





## 一一对应的moe（moe2）：
保存在exp6中
### 修改：
1、transformer：
增加MoEFFN、MoEDeformableTransformerDecoderLayer，MOE_DeformableTransformerDecoder

forward加入传回moe_expert_indices
2、head:
增加MOE_RTDETRDecoder类实现新功能
接收transformer的moe_expert_indices
返回接收transformer的moe_expert_indices
3、nn/tasks：
添加接收head中传回来的exports，修改loss方法中的损失计算

4、ultralytics/models/utils/loss.py
增加了moe_loss的修改

### 测试结果：
结果保存在runs_origin/val_val/moe2_1
map：33.6
ap50：53.5
GFLOPs 161.4 
![](image/Pasted%20image%2020260202122909.png)

uavdetr-r50 summary: 565 layers, 47381822 parameters, 0 gradients, 161.4 GFLOPs

## 文件说明：
保存图片条件的txt文件每个文件一行，
每行代表的意义：是否低光(0=无/1=是)	是否雾天(0=无/1=是)	目标是否密集(0=无/1=是)	目标是否空旷(0=无/1=是)
### 查看loss：
tensorboard --logdir=./output/visdrone/result1/tensorboard --port=6006