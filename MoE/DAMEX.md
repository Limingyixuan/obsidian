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

python -m torch.distributed.launch --nproc_per_node=4 main.py       --config_file config/uavdet8/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=40 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets uavdet_1 uavdet_2 uavdet_3 uavdet_4 

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


指令：
python -m torch.distributed.launch --nproc_per_node=4 main_change.py       --config_file config/uavdet8/DAMEX_4scale.py       --damex       --options batch_size=2 save_checkpoint_interval=4 epochs=40 lr=0.00014       --output_dir ./output/uodb/expt       --coco_path data/       --datasets uavdet_1 uavdet_2 uavdet_3 uavdet_4      --dataset_file uav  



# class MOELayer_change(torch.nn.Module):

#     """Tutel optimized MOELayer

#     """

#     # 设置每张卡上放多少export

#     @staticmethod

#     def global_expert_count(num_local_experts, group=None):

#         if not isinstance(num_local_experts, int):

#             num_local_experts = -int(1 / (num_local_experts + 1e-5))

#         world_size = C.get_world_size(group)

#         if num_local_experts == 0:

#             raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)

#         if num_local_experts > 0:

#             return num_local_experts * world_size

#         assert world_size % -num_local_experts == 0, f"Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."

#         return world_size // -num_local_experts

  

#     # 加载权重权重

#     def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):

#         buff_name = prefix + '_num_global_experts'

#         if buff_name not in state_dict:

#             logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")

#             logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")

#             logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")

#             state_dict[buff_name] = self._num_global_experts

#         else:

#             state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts

#             assert state_experts == expect_experts, "Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

  

#         for name, param in self.experts.named_parameters():

#             buff_name = prefix + 'experts.' + name

#             try:

#                 assert buff_name in state_dict, "Could not find parameter `%s` in state_dict." % buff_name

#                 if state_dict[buff_name].numel() == param.numel():

#                     state_dict[buff_name] = state_dict[buff_name].view(param.shape)

#             except:

#                 print("Make sure you are loading as pre-train model")

#         return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  

#     def state_dict(self, destination=None, prefix='', keep_vars=False):

#         return super().state_dict(destination, prefix, keep_vars)

  

#     @property

#     def num_global_experts(self):

#         return int(self._num_global_experts)

  

#     def __init__(

#         self,

#         gate_type,

#         model_dim: int,

#         experts=None,

#         scan_expert_func=None,

#         result_func=None,

#         group=None,

#         seeds=None,

#         a2a_ffn_overlap_degree=1,

#         is_postscore=True,

#         batch_prioritized_routing=False,

#         normalize_gate=True,

#         is_gshard_loss=True,

#         parallel_type='auto',

#         use_2dh=False,

#         damex_loss = True,

#         # 阈值

#         infer_threshould = 0.5,

  
  
  

#         **kwargs,

  

#     ):

#         super().__init__()

#         # 要求特征维度为偶数

#         assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim

#         group = group or dist.group.WORLD # 确定MoE的分布式通信范围

  

#         # 废弃参数处理，清理无效参数

#         if 'pad_samples' in kwargs:

#             logging.warning(f"`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.")

#             kwargs.pop('pad_samples')

#         for k in kwargs:

#             raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)

  

#         # 保存分布式通信组，后续跨卡通信使用

#         self.group = group

#         self.result_func = result_func # 结果处理函数，对MoE输出做

#         self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0) # 是否跳过MoE

  
  

#         # 确定单节点的专家数

#         self.num_local_experts = experts.pop('count_per_node', 1)

#         self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))

  

#         self.world_size = C.get_world_size(self.group)

#         if self.num_global_experts < self.world_size:

#             self.sharded_count = self.world_size // self.num_global_experts

#             self.num_local_experts = 1

#         else:

#             self.sharded_count = 1

  

#         self.force_data_parallel, self.force_adaptive, self.adaptive_degree = False, False, self.sharded_count

#         if parallel_type.startswith('adaptive:'):

#             self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])

#             if self.adaptive_degree == 0:

#                 self.force_data_parallel = True

#             else:

#                 if self.adaptive_degree < 0 or self.sharded_count % self.adaptive_degree != 0:

#                     valids = [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]

#                     raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, valids))

#                 self.force_adaptive = True

#             self.auto_parallel, self.use_model_parallel = False, True

#         elif self.sharded_count == 1:

#             self.auto_parallel, self.use_model_parallel = False, False

#         elif parallel_type in ('data', 'model'):

#             self.auto_parallel, self.use_model_parallel = False, (parallel_type == 'model')

#         elif parallel_type == 'auto':

#             self.auto_parallel, self.use_model_parallel = True, False

#         else:

#             raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

  

#         self.model_dim = model_dim

  

#         self.is_postscore = is_postscore

#         self.batch_prioritized_routing = batch_prioritized_routing

#         if int(os.environ.get('BATCH_PRIO', 0)) != 0:

#             self.batch_prioritized_routing = True

#         self.normalize_gate = normalize_gate

#         self.is_gshard_loss = is_gshard_loss

  

#         self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree

#         self.use_2dh = use_2dh

  

#         if seeds is not None and seeds[1] is not None:

#             torch.manual_seed(seeds[1])

  

#         experts_type = experts.pop('type')

#         if experts_type == 'custom':

#             self.experts = cast(ModuleList, experts['module'])

#         else:

#             assert re.match(r'[a-zA-Z0-9\_]+', experts_type), "Expert type must only include digits, letters and underline characters."

#             try:

#                 fused_experts = importlib.import_module(f'tutel.experts.{experts_type}', __name__)

#             except ModuleNotFoundError:

#                 raise Exception('Builtin expert type is not recognized: %s' % experts_type)

  

#             if experts_type == 'ffn':

#                 assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."

#                 assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

  

#             self.experts = fused_experts.ExpertModule(**experts)

  

#         self.experts.update(self)

  

#         if scan_expert_func is not None:

#             for n, p in self.experts.named_parameters():

#                 scan_expert_func(n, p)

#         for n, p in self.experts.named_parameters():

#             setattr(p, '_tutel_expert', True)

  

#         if isinstance(gate_type, str):

#             assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type

#             top_k = int(gate_type[3:-4])

#             logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")

#             gate_type = {'type': 'top', 'k': top_k}

  

#         if not isinstance(gate_type, list):

#             gate_type = [gate_type]

  

#         self.gates = []

#         for gi, single_gate_type in enumerate(gate_type):

#             gate_type = single_gate_type['type']

#             single_gate_type.pop('type')

#             assert re.match(r'[a-zA-Z0-9\_]+', gate_type), "Gate type must only include digits, letters and underline characters."

  

#             if seeds is not None and seeds[0] is not None:

#                 torch.manual_seed(seeds[0] + gi)

#             try:

#                 single_gate = importlib.import_module(f'tutel.gates.{gate_type}', __name__)

#             except ModuleNotFoundError:

#                 raise Exception("Unrecognized gate_type: %s" % gate_type)

  

#             gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, **single_gate_type)

#             if not hasattr(gate_module, 'gate_noise'):

#                 gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)

#             if not hasattr(gate_module, 'capacity_factor'):

#                 gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

  

#             self.gates += [gate_module]

  

#         self.gates = ModuleList(self.gates)

#         self.damex_loss = damex_loss

  

#         if seeds is not None and len(seeds) > 2 and seeds[2] is not None:

#             torch.manual_seed(seeds[2])

  

#     def extra_repr(self):

#         return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % (

#             [f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates],

#             self.num_global_experts,

#             self.world_size,

#         )

  

#     def get_parameter_iterator(self, param_type):

#         if param_type == 'gate':

#             return self.gates.named_parameters()

#         elif param_type == 'local_experts':

#             return self.experts.named_parameters()

#         else:

#             raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

  

#     def expert_local(self, x, reserve_shape):

#         y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)

#         self.protected_shape = y.shape

#         return y.reshape(y.size(0), y.size(1), -1)

  

#     def forward(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=True):

#         if self.skip_moe: # 判断是否跳过MoE

#             result_output = input

#             result_output.l_aux = None

#             return self.result_func(result_output) if self.result_func is not None else result_output

  

#         original_shape, original_dtype  = input.shape, input.dtype

#         assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"

  

#         x = input.reshape(-1, original_shape[-reserve_dims:].numel()) # 转变为二维张量，方便后续的路由计算

#         for p in self.experts.parameters():

#             x = x.to(p.dtype)

#             break

#         gctx = self.gates[gate_index]

#         a2a_ffn_overlap_degree = a2a_ffn_overlap_degree if a2a_ffn_overlap_degree is not None else self.a2a_ffn_overlap_degree

#         self.moe_loss_fn = None

  

#         # 路由函数

#         def routing():

#             # 计算初始得分

#             logits = gctx(x)

  

#             # 训练时添加噪声

#             if self.training and gctx.gate_noise > 0:

#                 logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts

#             else:

#                 logits_w_noise = logits

  

#             # 计算概率

#             scores = F.softmax(logits_w_noise, dim=1)

#             '''

#             # for debuggin

#             global file_count

#             selection = scores #.argmax(-1)

#             gt = scores.clone().detach().cpu().numpy()

#             dir_path = '../../jsr_all-coco-0.01/epoch_11/experts'

#             os.makedirs(dir_path, exist_ok=True)

#             with open(f'{dir_path}/{file_count}.pkl', 'wb') as f:

#                 pkl.dump(gt,f)

#                 f.close()

#             file_count += 1

#             '''

#             # 定义MOE损失函数

#             if self.is_gshard_loss:

#                 _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)

#             else:

#                 _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(

#                     F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),

#                     self.num_global_experts, gctx.gate_noise) #NOTE: gates is not being used here what a bs!

#             # 选择top-k个专家

#             # if not self.training:

#             #     batch_size = input.shape[0]

#             #     token_per_sample = input.shape[1] if len(input.shape) >=3 else 1

#             #     target_experts_list = []

#             # for sample_idx in range(batch_size):

#             #     # 取当前样本所有token的平均概率（代表样本整体场景特征）

#             #     sample_scores = scores[sample_idx*token_per_sample : (sample_idx+1)*token_per_sample].mean(dim=0)

#             #     # 筛选概率超阈值的专家

#             #     expert_ids = torch.where(sample_scores > self.infer_threshold)[0].cpu().numpy().tolist()

#             #     # 适配文档复用场景：

#             #     if len(expert_ids) >=2:  # 多专家超阈值→复合场景（功能复用）

#             #         expert_ids.append(self.fusion_expert_idx)  # 强制添加E4融合

#             #         expert_ids = list(set(expert_ids))  # 去重

#             #     elif len(expert_ids) ==1:  # 单专家超阈值→难场景（参数复用）

#             #         pass  # 仅激活专属专家，不添加E4

#             #     else:  # 无专家超阈值→激活通用专家E4

#             #         expert_ids = [self.fusion_expert_idx]

#             #     target_experts_list.append(expert_ids)

  
  
  
  
  
  

#             topk_ids = torch.topk(scores, gctx.top_k, dim=1).indices

  

#             # fg_mask moe loss function

#             # 基于掩码的损失函数

#             if self.is_gshard_loss:

#                 self.moe_loss_fn = lambda fg_mask, buf : losses.gshard_loss(logits[fg_mask], topk_ids[fg_mask])

#             else:

#                 self.moe_loss_fn = lambda fg_mask, buf : losses.load_importance_loss(

#                     F.softmax(logits[fg_mask, :], dim=1), logits.gather(index=topk_ids, dim=1)[fg_mask, :],

#                     self.num_global_experts, gctx.gate_noise)

#                 # if dist.get_rank() == 0:  # or another rank you want to debug

#                 #     breakpoint()

#                 # else:

#                 #     torch.distributed.barrier()

  

#                 # self.damex_loss_fn = lambda fg_mask, dataset_targets: F.cross_entropy(logits[fg_mask], dataset_targets[fg_mask] ,label_smoothing=0) if dataset_targets is not None else self.moe_loss_fn(fg_mask)

#                 self.damex_loss_fn = lambda fg_mask, dataset_targets: F.cross_entropy(logits, dataset_targets ,label_smoothing=0) if dataset_targets is not None else self.moe_loss_fn(fg_mask, None)

  

#             # if dist.get_rank() == 0:  # or another rank you want to debug

#             #     breakpoint()

#             # else:

#             #     torch.distributed.barrier()

#             # 提取路由关键信息

#             #     return (num_global_experts, indices_s, locations_s, gates_s, capacity), l_loss

#             # 即(全局专家的总数量，每个top-k个位置的专家索引列表，在发到对应专家后所处的位置，样本分配给对应专家的权重，每个专家的处理容量上限)

#             # gates_s就是每个样本对于每个专家的权重

#             return logits.dtype, extract_critical(scores,#每个样本对专家计算结果归一化后的得分

#                 top_k = gctx.top_k if top_k is None else top_k, # 要选择的top-k个专家的数量

#                 loss_fn = _loss_fn,# MoE损失函数用于负载均衡

#                 capacity_factor = gctx.capacity_factor if capacity_factor is None else capacity_factor,# 专家容量

#                 batch_prioritized_routing = self.batch_prioritized_routing,#

#                 normalize_gate = self.normalize_gate,

#                 group = self.group,

#                 alignment = self.sharded_count * a2a_ffn_overlap_degree,# 容量对其因子

#                 inequivalent_tokens = inequivalent_tokens,# 是否适配多卡样本数不用的场景

#             )

#             # (

#             #   logits.dtype,  # 第一部分：门控得分的数值类型

#             # (crit, l_aux)  # 第二部分：extract_critical 返回的二元组

  
  

#         if x.is_cuda:

#             with torch.cuda.amp.autocast(enabled=False):

#                 logits_dtype, (crit, l_aux) = routing()

#         else:

#             logits_dtype, (crit, l_aux) = routing()

#         #

#         y = fast_encode(x.to(logits_dtype), crit, self.is_postscore).to(x.dtype)

  

#         # if dist.get_rank() == 0:  # or another rank you want to debug

#         #     breakpoint()

#         # else:

#         #     torch.distributed.barrier()

  
  

#         if self.force_data_parallel:

#             y = self.expert_local(y, original_shape[-reserve_dims:])

#         else:

#             if self.auto_parallel:

#                 self.use_model_parallel = (y.numel() * (self.sharded_count - 1) * 2 < sum([x.numel() for x in self.experts.parameters()]))

  

#             if self.num_global_experts < self.world_size:

#                 if self.use_model_parallel:

#                     y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))

#                 else:

#                     y = y.view(self.world_size, -1, y.size(2))

  

#             if a2a_ffn_overlap_degree > 1 and y.is_cuda:

#                 def expert_fn(expert_input):

#                     return self.expert_local(expert_input, original_shape[-reserve_dims:])

#                 y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)

#             else:

#                 y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)

#                 y = self.expert_local(y, original_shape[-reserve_dims:])

#                 y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)

  

#             if self.num_global_experts < self.world_size:

#                 if self.use_model_parallel:

#                     y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)

#                 else:

#                     y = y.view(self.num_global_experts, -1, y.size(2))

  

#         y = fast_decode(y.to(logits_dtype), crit, self.is_postscore)

  

#         y = y.view(list(original_shape[:-reserve_dims]) + list(self.protected_shape[-reserve_dims:])).to(original_dtype)

#         self.l_aux = y.l_aux = l_aux

  

#         # 返回了两个元素，第一个时MoE的输出张量，第二个是损失函数

#         return self.result_func(y) if self.result_func is not None else y, self.damex_loss_fn if self.damex_loss else self.moe_loss_fn