从DETR来的
transformer的架构:
![](Pasted%20image%2020250718163933.png)
左边是编码器,右边是解码器.
编码器由N个堆叠而成.
每个编码器有两个子层,第一个子层包括一个多头注意力层和规范化层以及一个残差连接层.第二个子层包括一个前馈全连接子层和规范化层和一个残差连接.
![](Pasted%20image%2020250718164446.png)

分词器和one-hot
分词器在一维，one-hot在多维。各有优劣。

![](Pasted%20image%2020250721101417.png)



### 过往模型


编码器解码器：

![](Pasted%20image%2020250721104201.png)

1、遗忘问题。
2、词之间的关联程度。


注意力机制：

![](Pasted%20image%2020250721104414.png)

解决词之间的关联程度的问题。
还存在串行化计算的问题，即无法并行计算。

### transformer
transformer使用了注意力机制但是完全舍弃了RNN，完全基于自注意力机制。
![](Pasted%20image%2020250718163933.png)

步骤：词嵌入，位置编码
 

![](Pasted%20image%2020250721105758.png)

512，每一维代表一个特征

多头注意力，将512分为8个64

##### 自注意力
先计算与不同输入的关联性

![](Pasted%20image%2020250721151739.png)

再用上一步得到的softmax的权重乘上各自的V矩阵就可以得到输出。

![](Pasted%20image%2020250721151942.png)

Q是一组查询语句，V是数据库，里面有若干数据项。对于每一条查询语句，我们期望从数据库中查询出一个数据项（加权过后的）来。如何查询？这既要考虑每个q本身，又要考虑V中每一个项。如果用K表示一组钥匙，这组钥匙每一把对应V中每一项，代表了V中每一项的某种查询特征，（所以K和V的数量一定是相等的，维度则没有严格限制，做attention时维度和q一样只是为了在做点积时方便，不过也存在不用点积的attention）。然后对于每一个Q中的q，我们去求和每一个k的attention，作为对应value的加权系数，并用它来加权数据库V中的每一项，就得到了q期望的查询结果。

所以query是查询语句，value是数据项，key是对应每个数据项的钥匙。名字起得是很生动的。不过和真正的数据库查询不一样的是，我们不仅考虑了查询语句，还把数据库中所有项都加权作为结果。所以说是全局的。

### masked self attention ：
掩码注意力机制。
目的是进行训练的生成任务时，生成单词时是一个一个生成的，


##### 多头注意力机制：
![](Pasted%20image%2020250721154831.png)






### 文件转编码的过程：




### 代码详解：
模型初始化：
```
class Transformer(nn.Module):  
    ''' A sequence to sequence model with attention mechanism. '''  
	  //
		初始化- `n_src_vocab` 和 `n_trg_vocab` 分别表示源语言和目标语言的词汇表大小。
		`src_pad_idx` 和 `trg_pad_idx` 是源语言和目标语言的填充符号的索引。
		d_word_vec=512：词向量的维度
		d_model：模型维度
		d_inner：前馈神经网络（全连接层？）的隐藏层维度
		n_layers=6：编码器和解码器堆叠的个数
		n_head=8,：多头注意力机制的头的数目
		d_k=64，d_v=64：键和值的维度
		n_position:位置编码的最大位置数。

		//
    def __init__(  
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,  
            d_word_vec=512, d_model=512, d_inner=2048,  
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,  
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,  
            scale_emb_or_prj='prj'):  
  
        super().__init__()  
  
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx  
  
       
        assert scale_emb_or_prj in ['emb', 'prj', 'none']  
		### scale_emb_or_prj的值必须是这三个里的其中给之一

		# `scale_emb` 是一个布尔值，用于决定是否对嵌入层的输出进行缩放。
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False  
        
        # - `scale_prj` 是一个布尔值，用于决定是否对投影层的输出进行缩放。
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False 
         
        self.d_model = d_model  


		#编码器和解码器初始化。
        self.encoder = Encoder(  
            n_src_vocab=n_src_vocab, n_position=n_position,  
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,  
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,  
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)  
  
        self.decoder = Decoder(  
            n_trg_vocab=n_trg_vocab, n_position=n_position,  
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,  
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,  
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)  



		#用于将 Transformer 解码器的输出映射到目标词汇表空间 
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)  
  
        for p in self.parameters():  
            if p.dim() > 1:  
                nn.init.xavier_uniform_(p)   
  
        assert d_model == d_word_vec, \  
        'To facilitate the residual connections, \  
         the dimensions of all module outputs shall be the same.'  
        if trg_emb_prj_weight_sharing:  
            # Share the weight between target word embedding & last dense layer  
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight  
  
        if emb_src_trg_weight_sharing:  
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight  
  
  
    def forward(self, src_seq, trg_seq):  
  
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)  
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)  
  
        enc_output, *_ = self.encoder(src_seq, src_mask)  
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)  
        seq_logit = self.trg_word_prj(dec_output)  
        if self.scale_prj:  
            seq_logit *= self.d_model ** -0.5  
  
        return seq_logit.view(-1, seq_logit.size(2))
```




![](Pasted%20image%2020250722102632.png)