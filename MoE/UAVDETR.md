
## 初始模型：
#### 训练1：291epoch：
![](image/Pasted%20image%2020260121094325.png)

### 400epoch：
#### 测试集：
AP50:0.421
mAP:0.25
![](image/Pasted%20image%2020260123093546.png)


#### 验证集：
AP50：52.5
mAP：32.9
![](image/Pasted%20image%2020260123140902.png)



## 加上初始moe：
#### 测试集
AP50:0.25
mAP:0.422
![](image/Pasted%20image%2020260122094245.png)

#### 验证集
AP50：53.3
mAP：33.5
![](image/Pasted%20image%2020260123141142.png)


## 一一对应的moe：
修改之处：
1、transformer：
增加MoEFFN、MoEDeformableTransformerDecoderLayer
修改DeformableTransformerDecoder
forward加入传回moe_expert_indices
2、head:
接收transformer的moe_expert_indices
返回接收transformer的moe_expert_indices
3、tasks：
添加接收head中传回来的exports，修改loss方法中的损失计算

4、


### 查看loss：
tensorboard --logdir=./output/visdrone/result1/tensorboard --port=6006