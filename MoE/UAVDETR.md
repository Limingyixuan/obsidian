
## 初始模型：
#### 训练1：291epoch：
![](image/Pasted%20image%2020260121094325.png)

### 400epoch：
runs_origin/train/exp

uavdetr-r50 summary: 544 layers, 44640370 parameters, 0 gradients, 
151.9 GFLOPs
#### 测试集：
AP50:0.421
mAP:0.25
![](image/Pasted%20image%2020260123093546.png)


#### 验证集：
151.9 GFLOPs
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

## visdrone按场景划分：
1、低光
uavdetr：
![](image/Pasted%20image%2020260205150025.png)
51.8
31.8
2、雾天：
![](image/Pasted%20image%2020260205150829.png)
55
33


3、小目标
![](image/Pasted%20image%2020260205150953.png)

52.5
32.6


4、正常目标：
![](image/Pasted%20image%2020260205151037.png)
60.4
40.9


moe2：
1、
![](image/Pasted%20image%2020260206153500.png)
53.5
32.9

2、
56.6
34.4
![](image/Pasted%20image%2020260206153607.png)


3、
52.6
33
![](image/Pasted%20image%2020260206153729.png)

4、
61.6
42.3

![](image/Pasted%20image%2020260206153820.png)




## 多类型标签统计：
--- 1. 单个标签统计 ---
低光 (Low Light):    1612 张
雾天 (Foggy):        757 张
多小目标 (Many Small): 3828 张
少小目标 (Few Small):  2643 张

--- 2. 标签组合统计 (格式: [低光,雾天,多小,少小] -> 数量) ---
[0, 0, 0, 1] -> 1564 张 (少小目标)
[0, 0, 1, 0] -> 2631 张 (多小目标)
[0, 1, 0, 1] ->  146 张 (雾天+少小目标)
[0, 1, 1, 0] ->  518 张 (雾天+多小目标)
[1, 0, 0, 1] ->  901 张 (低光+少小目标)
[1, 0, 1, 0] ->  618 张 (低光+多小目标)
[1, 1, 0, 1] ->   32 张 (低光+雾天+少小目标)
[1, 1, 1, 0] ->   61 张 (低光+雾天+多小目标)