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


## 航拍数据集：
- 专家 1：处理「航拍大目标」（DOTA、FAIR1M）；
- 专家 2：处理「低空小目标」（VisDrone、UAVDT）；
- 专家 3：处理「夜间红外目标」（FLIR UAV、KAIST UAV）；

测试指令：
python -m torch.distributed.launch --nproc_per_node=8 main.py       --config_file config/uodb/DAMEX_4scale.py       --damex       --options batch_size=1 save_checkpoint_interval=4 epochs=36 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets coco dota hazydet uavdet visdrone uav_dark wideface

第一次测试：
数据集：
1、coco：coco格式
2、dota：coco格式
3、hazdet:coco格式
4、uavdet：coco格式
5、visdrone：coco格式
6、uav_dark:coco格式
7、soda：coco格式
8、tiny_person: