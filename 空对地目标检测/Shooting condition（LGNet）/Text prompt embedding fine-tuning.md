代码理解：
## 运行函数：
```
if __name__ == "__main__":
	main()  
  
  
	# ''' register forward hook CustomCLIP forward text_features '''  
	output_text_feats = []  
	def hook(module, input, output):  
	    output_text_feats.append(output)  
	  
	model, preprocess = clip.load("ViT-B/32", device="cuda:0")  
	custom_clip = CustomCLIP(model)  
	# if have checkpoints  
	# custom_clip.prompt_learner.load_state_dict(torch.load("models/clip_29.pth"))  
	custom_clip.text_encoder.register_forward_hook(hook)  
	custom_clip.eval()  
	image = torch.randn(1,3,224,224).cuda()  
	logits = custom_clip(image)  
	  
	save_text_feats = output_text_feats[0]  
	fix_text_feats = torch.load("text_learner/text_feats.pt")  
	save_text_feats = (save_text_feats + fix_text_feats) / 2  
	torch.save(save_text_feats,"text_learner/uavdt_text_feats.pth")
```

代码的主函数：
main函数进行训练。


```
class PromptLearner(nn.Module):  
    def __init__(self,clip_model,prompt_template:list,learning_pos:list,learning_counts:list) -> None:  
          
        super().__init__()  
  
        prompt = clip.tokenize(prompt_template).cuda()  
        # prompt是词汇表索引（3 * 77）
        with torch.no_grad():  
            embedding = clip_model.token_embedding(prompt)  
		  #语义向量：（3，77，512）
        hight_prompt = ['high','medium','low']  
        angle_prompt = ['bird','side','front','other']  
  
        hight_prompt = clip.tokenize(hight_prompt).cuda()  
        angle_prompt = clip.tokenize(angle_prompt).cuda()  
        with torch.no_grad():  
            hight_embedding = clip_model.token_embedding(hight_prompt)  
            angle_embedding = clip_model.token_embedding(angle_prompt)  
  
        # shape = [n_learning_vectors, embedding_dim]  
        learn_vec_list = nn.ParameterList()  
        self.counts = 1  
        for idx , counts in enumerate(learning_counts):  
            # learn_vec_list.append(nn.Parameter(learning_vectors[idx].repeat(counts,1)))  
            if idx == 0:  
                learn_vec_list.append(nn.Parameter(hight_embedding[:,1]))  
            else:  
                learn_vec_list.append(nn.Parameter(angle_embedding[:,1]))  
            self.counts *= counts  
        self.learn_vec_list = learn_vec_list  
  
        self.pos = learning_pos  
          
  
        self.freeze_embedding = embedding  
  
        self.tokenized_prompts = prompt[0]
```
作用：管理可学习的提示向量
prompt = clip.tokenize(prompt_template).cuda()  ：返回pytorch张量表示文本字符串的标识
这里的3表示三句话，也就是白天黑夜和雾天三种情况，即同时处理的文本数量
77表示将一句话转皇城token后，一句话最多的token数量
512是将token向量化后的特征数量
==在将词转化为token的编码表示时，每个词的向量表示，不仅会有此本身的序列值，同样在结尾和开头会加上表示序列首位的固定ID==


```
class CustomCLIP(nn.Module):  
    def __init__(self,clip_model) -> None:  
        super().__init__()  
        prompt_list = [  
             "A alpha altitude beta view of a foggy day taken by a drone.",  
             "A alpha altitude beta view of a night day taken by a drone.",  
             "A alpha altitude beta view of a sunny day taken by a drone.",  
  
            # "foggy view from drone at alpha altitude and beta angle .",  
            # "night view from drone at alpha altitude and beta angle .",            # "daylight view from drone at alpha altitude and beta angle ."        ]  
  
        learning_pos = [2,4]  
        learning_counts = [3,4]  
        self.prompt_learner = PromptLearner(clip_model,prompt_list,learning_pos,learning_counts)  
  
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  
        self.image_encoder = clip_model.visual  
        self.text_encoder = TextEncoder(clip_model)  
        self.logit_scale = clip_model.logit_scale  
        self.dtype = clip_model.dtype  
          
    def forward(self, image):  
        image_features = self.image_encoder(image.type(self.dtype))  
  
        prompts = self.prompt_learner().type(self.dtype)  
        tokenized_prompts = self.tokenized_prompts  
        text_features = self.text_encoder(prompts, tokenized_prompts)  
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
  
        logit_scale = self.logit_scale.exp()  
        logits = logit_scale * image_features @ text_features.t()  
  
        return logits
```


- `prompt_list`：包含3个天气条件模板（雾天/夜晚/晴天），模板中的`alpha`和`beta`为可学习参数。

- `learning_pos`：指定模板中可学习参数的位置（索引2和4对应`alpha`和`beta`）。