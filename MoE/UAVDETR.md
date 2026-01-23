
### 初始模型：
#### 训练1：291epoch：
![](image/Pasted%20image%2020260121094325.png)

400epoch：
AP50:0.421
mAP:0.25
![](image/Pasted%20image%2020260123093546.png)


加上初始moe：
AP50:0.25
mAP:0.422
![](image/Pasted%20image%2020260122094245.png)

### 查看loss：
tensorboard --logdir=./output/visdrone/result1/tensorboard --port=6006