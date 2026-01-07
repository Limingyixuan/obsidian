组合数据集有利于提高对未知环境的感知能力

困难：
域变换
数据不平衡

无需手动干预、无需统一标签、无参数冗余、测试时领域无关

使用基于混合专家（MoE）[14]的目标检测方法

基于DINO框架来构建方法


##  测试时 “自动分诊”—— 不用人工判断数据来源，路由器自主决策

测试时，输入图像没有任何 “数据集标记”，但模型能自动适配：

- 输入一张图（比如路边既有车辆又有交通标志），路由器会先分析图像特征：“这张图里有 COCO 常见的车辆特征，也有 LISA 常见的交通标志特征”；
- 路由器会自动把 “车辆相关的特征” 送到专家 1，“交通标志相关的特征” 送到专家 2；
- 两个专家分别处理后，把结果传给共享检测头，检测头直接输出 “车辆 + 交通标志” 的完整结果 —— 全程不用人工干预 “这张图来自哪个数据集”“该用哪个检测器”。




## 原始数据集：

- Pascal VOC：
- MS-COCO：
- KITTI
- WiderFace
- DOTA
- DeepLesion
- Clipart
- Comic
- Watercolor
- LISA
- Kitchen


环境安装：
安装dino时记得切换到11.8cuda
pip install pycocotools



环境配置：```
python -m pip install panopticapi --index-url=https://pypi.tuna.tsinghua.edu.cn/simple 
```

## 航拍数据集：
- 专家 1：处理「通用目标」（coco）；
- 专家 2：处理「遥感目标」（dota）；
- 专家 3：处理「雾天目标」（hazdet）；
- 专家4：处理「多拍摄条件」（UAVDT）
- 专家5：处理「多类别」（UAVDT）
- 专家6：处理「黑夜」（uav_dark）
- 专家7：处理「小目标」（soda）
- 专家8：处理「海边-人群」（tiny-person）
测试指令：
python -m torch.distributed.launch --nproc_per_node=8 main.py       --config_file config/uodb/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=36 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets coco dota hazydet uavdet visdrone uav_dark soda tiny

6个数据集：
CUDA_VISIBLE_DEVICES=3,4,5,6,7,8 
python -m torch.distributed.launch --nproc_per_node=6 main.py       --config_file config/uodb/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=36 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets coco hazydet uavdet visdrone uav_dark tiny


4个
CUDA_VISIBLE_DEVICES=6,7,8,9
python -m torch.distributed.launch --nproc_per_node=4 main.py       --config_file config/uodb/DAMEX_4scale.py       --damex       --options batch_size=1 save_checkpoint_interval=4 epochs=36 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets dota hazydet uavdet visdrone --nproc_per_node=2 --amp



```bash
CUDA_VISIBLE_DEVICES=3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=6 main.py \
    --config_file config/uodb/DAMEX_4scale.py \
    --damex \
    --options batch_size=2 save_checkpoint_interval=4 epochs=36 lr=0.00014 \
    --output_dir ./output/uodb/expt \
    --datasets hazydet uavdet visdrone uav_dark soda tiny
```


```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 main.py \
    --config_file config/uodb/DAMEX_4scale.py \
    --damex \
    --options batch_size=1 save_checkpoint_interval=4 epochs=36 lr=0.00014 \
    --output_dir ./output/uodb/expt \
    --coco_path data/ \
    --datasets coco dota hazydet uavdet visdrone uav_dark soda tiny
```
export NCCL_P2P_DISABLE=1


### 测试
#### 第一次测试：
数据集：
- 1、coco：coco格式
- 2、dota：coco格式
- 3、hazdet:coco格式
- 4、uavdet：coco格式
- 5、visdrone：coco格式
- 6、uav_dark:coco格式
- 7、soda：coco格式
- 8、tiny_person:coco格式
问题一：
一直出现显存不足的问题：

初步推测是数据集的问题，

问题二：
gcc版本太高：
命令：
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH


#### 第二次测试，用两个数据集：
coco + visdrone
python -m torch.distributed.launch --nproc_per_node=2 main.py       --config_file config/uodb/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=36 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets coco visdrone

GPU情况：
image/Pasted%20image%2020251105092844.png)

![](image/Pasted%20image%2020251105093952.png)

第二次测试

python -m torch.distributed.launch --nproc_per_node=8 main.py       --config_file config/uavdet8/DAMEX_4scale.py       --damex       --options batch_size=1 save_checkpoint_interval=4 epochs=12 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets uavdet_1 uavdet_2 uavdet_3 uavdet_4 uavdet_5 uavdet_6 uavdet_7 uavdet_8

推理代码：
多GPU推理：
```bash
torchrun --nproc_per_node=8 infer.py \  # 8个GPU
  --config_file ./configs/damex_uodb.py \
  --checkpoint_path ./output/checkpoint_best_regular.pth \
  --input_path ./test_images/ \
  --output_dir ./infer_results_multi_gpu \
  --dist_url tcp://127.0.0.1:23456 \  # 分布式通信URL
  --amp \
  --save_vis
```



第三次测试：

python -m torch.distributed.launch --nproc_per_node=4 main2.py       --config_file config/uavdet8/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=40 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets uavdet_1 uavdet_2 uavdet_3 uavdet_4 

数据划分：
![](image/Pasted%20image%2020251120184935.png)

export1：daylight（low and medium）

export2：night

export3：hight-alt

export4：fog


12 epoch2：
![](image/Pasted%20image%2020251120184837.png)

4 epochs：

![](image/Pasted%20image%2020251121152009.png)


测试指令：

python -m torch.distributed.launch --nproc_per_node=4 main_test.py       --config_file config/uavdet8/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=12 lr=0.00014       --output_dir ./output/uodb/expt      

### 可添加：
来自UAVDETR的MSFF-FE模块：
作用是：通过组合跨多个尺度的空间和频域信息来保留小目标细节。
![](image/Pasted%20image%2020251112205051.png)
MSFF-FE：

![](image/Pasted%20image%2020251112205102.png)
FOCU模块：


![](image/Pasted%20image%2020251112205122.png)



将三通道的图片进行4 * 4的分割，得到12通道的特征

之后与其他尺度的特征图fous之后的特征拼接起来，经过split分成两部分，
其中一部分





第四次实验：模型修改：

实现，复合场景使用多个专家，自动匹配对应的几个专家

数据集：
dota：基准，待定
visdrone
uavdet：
hazydet
uavdark

增加阈值参数：small_obj_thr、low_light_thr、fog_contrast_thr，可对图片是否属于小目标/低光/雾天进行判断

使用大模型对图片的属性进行判断通义千问大模型
配置环境变量：
```bash
echo "export DASHSCOPE_API_KEY='sk-4326c37b9758438793dce16f90865a90'" >> ~/.bashrc
```

token计算：
uavdet数据集：resize为1024 * 1024：28 * 28的图像为1token所以转变后的一张图片的token是1369，50753张图片结果为55,766,215token 55块钱 
batch调用半价即，27.5元

43.47 - 2780


对于门控单元的修改：
```python
def forward(self, input: Tensor, scene_labels=None, gate_index=0, ...):
    # 路由函数
    def routing():
        # 原生逻辑：gate_type计算专家得分（如cosine_top的余弦相似度→logits）
        logits = gctx(x)
        # 训练时加噪声，推理时不加
        if self.training and gctx.gate_noise > 0:
            logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        # 原生逻辑：计算专家选择概率（softmax）
        scores = F.softmax(logits_w_noise, dim=1)

        # 新增：推理时阈值筛选专家（核心逻辑）
        if not self.training:
            batch_size = input.shape[0]
            token_per_sample = input.shape[1] if len(input.shape) >=3 else 1
            target_experts_list = []
            for sample_idx in range(batch_size):
                # 取当前样本所有token的平均概率（代表样本整体场景特征）
                sample_scores = scores[sample_idx*token_per_sample : (sample_idx+1)*token_per_sample].mean(dim=0)
                # 筛选概率超阈值的专家
                expert_ids = torch.where(sample_scores > self.infer_threshold)[0].cpu().numpy().tolist()
                
                # 适配文档复用场景：
                if len(expert_ids) >=2:  # 多专家超阈值→复合场景（功能复用）
                    expert_ids.append(self.fusion_expert_idx)  # 强制添加E4融合
                    expert_ids = list(set(expert_ids))  # 去重
                elif len(expert_ids) ==1:  # 单专家超阈值→难场景（参数复用）
                    pass  # 仅激活专属专家，不添加E4
                else:  # 无专家超阈值→激活通用专家E4
                    expert_ids = [self.fusion_expert_idx]
                
                target_experts_list.append(expert_ids)
            
            # 生成掩码：仅激活超阈值的专家，屏蔽其他专家
            logit_mask = torch.zeros_like(logits, device=logits.device)
            for token_idx in range(logits.shape[0]):
                sample_idx = token_idx // token_per_sample
                target_experts = target_experts_list[min(sample_idx, batch_size-1)]
                logit_mask[token_idx, target_experts] = 1.0
            logits = logits + (1 - logit_mask) * (-1e9)  # 屏蔽无关专家

        # 原生逻辑：选择Top-k专家（此时仅超阈值专家有概率，其余为0）
        topk_ids = torch.topk(scores, gctx.top_k, dim=1).indices
        # 其余损失计算、extract_critical逻辑不变...
        return logits.dtype, extract_critical(...)
    # 其余forward逻辑不变...
```

1、将标签输入到moe层，进行路由分配，方法是，使用标签中的sence（场景字段）当前有三个值【是否包含小目标，是否为黑夜，是否为雾天】


uav测试



指令：


export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.5,expandable_segments:True"
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501   main_change.py       --config_file config/uav_data/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=40 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets uavdet visdrone


### 显存泄露：
模型的大小：1076M

修改1：
修改export的



## 对比论文
### 1、SM3Det：
moe的uav目标检测，目标是解决多模态的目标检测任务，还有多任务（意思是不同的检测框，不同的拍摄角度）
对比点：sm3det没有对于多拍摄条件的优化，

### 2、UAV-DETR: 
Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery


### 3、Enhancing_RT_DETR



### 4、YOLO-Master
