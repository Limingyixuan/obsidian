
1、数据集构建
训练数据模板：

{ "id": "uav_hazy_001", // 无人机样本唯一ID "image": "hazy_uav_001.jpg", // 图像路径（对应脚本里的--image_folder） "conversations": [ { "from": "human", "value": "<image>\n请标注该无人机航拍图像的核心信息：\n1. 场景属性（天气：雾天/夜间/晴天；环境：城市/乡村/郊区）；\n2. 目标实例类别（汽车/行人/建筑/无人机等）；\n3. 场景感知结论（用于下游检测模型专家路由）。" }, { "from": "gpt", "value": "场景属性：雾天、城市道路；目标实例：汽车（3辆）、建筑（5栋）；场景感知结论：能见度低，需调用雾天专用检测分支。" } ] }

测试数据集构建：


```
from transformers import AutoModelForCausalLM, AutoProcessor 
import torch 
from peft import PeftModel 
# ===================== 你只需要改这 3 个路径 ===================== BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct" # 基座模型 LORA_PATH = "output/qwen3-vl-uav-lora" # 你的LoRA权重路径 TEST_IMAGE = "hazy_uav_001.jpg" # 测试无人机图片 # ================================================================= # 加载处理器 processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True) 
# 加载模型 + LoRA 
model = AutoModelForCausalLM.from_pretrained( 
BASE_MODEL, 
torch_dtype=torch.bfloat16, 
device_map="cuda", 
trust_remote_code=True ) model = PeftModel.from_pretrained(model, LORA_PATH) model = model.to("cuda") model.eval() # ===================== 【
测试输入】就是你训练时的 human 指令 ===================== 
prompt = """<image> 请标注该无人机航拍图像的核心信息： 
1. 场景属性（天气：雾天/夜间/晴天；环境：城市/乡村/郊区）；
2. 目标实例类别（汽车/行人/建筑/无人机等）； 
3. 场景感知结论（用于下游检测模型专家路由）。""" # 构建输入 
messages = [{"role": "user", "content": prompt}] 
text = processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True ) 
# 图像输入 
inputs = processor( text=[text], images=[TEST_IMAGE], return_tensors="pt" ).to("cuda") # 生成 
with torch.no_grad(): outputs = model.generate( **inputs, max_new_tokens=256, temperature=0.1, # 越低输出越固定、越标准 do_sample=False ) 
# 输出结果 
response = processor.decode(outputs[0], skip_special_tokens=True) print("="*50) print("模型输出：\n", response.split("assistant\n")[-1])
```


测试输出：
```
场景属性：雾天、城市道路；目标实例：汽车、建筑；

```
2、
