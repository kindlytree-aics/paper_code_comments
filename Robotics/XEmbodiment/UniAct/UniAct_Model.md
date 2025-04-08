# UniAct Model结构

## 模型的整体结构介绍，如何支持多模态？

回答：模型结构参考链接：[整体结构图](./uniact_arch.png)，主要由如下部分组成：
- 多模态大模型Backbone：VLM模型为`LlavaOnevisionForConditionalGeneration`，支持视觉和自然语言的多模态信息融合（多个模态数据不是进行cross attention计算而是在输入层进行拼接融合），

```
# initialize base model  ua = universal action
# .\models\UniAct_V1.py
self.ua_extractor = LlavaOnevisionForConditionalGeneration.from_pretrained(
    ua_extractor_name,
    torch_dtype='auto', 
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
    )
```
- `universal_actions_codebook`为universal action语义信息融合提取层，在这里定义了码本`codebook`（离散表示空间，如`codebook_size=64`则表示将动作元语建模为64个），基于`codebook_size=64`个latent action primitives的加权和输出最终的universal action embeddings。
,关于其实现原理如加权和的权重如何获取可以参考问题2的描述。

- action head，在代码实现中提供了两种head：
  - 将`universal action embeddings`和`vision_embedding`拼接作为三层MLP的输入输出特定的机器人action行为预测（`MLP_Decoder`）。具体的描述参考问题3。
  - 在上面的基础上输入中再多加入本体感知观测数据`proprios`（`ACT_Decoder`）。

## 问题2：如何实现通过线性加权latent action primitives得出universal action？

GumbelVQ类实现了latent action primitives进行加权后得出的综合action，即Universal Action(返回的quantized变量）。
其中`codebook_size`可以理解为`latent action primitives`的数量，该类将backbone输出的张量通过pre_proj线性层映射为codebook_size维度，然后通过`gumbel_softmax`得出归一化的加权系数（smooth label）。
同时`entropy_loss`定义了用KL loss来学习时代codebook_size大小的元语的多项式分布尽可能成均匀分布，具体解释可以参考[GumbelSoftmax](./GumbelSoftmax.md)

```
# .\UniAct\models\UniAct_V1.py
def forward(self, logits, temperature = None, hard_forward = False):
    
    if hard_forward: return self.greedy_forward(logits)
    temperature = temperature if temperature is not None else self.temperature
    # 输入 logits 经过一个线性变换（例如全连接层）以匹配 codebook 的大小
    logits = self.pre_proj(logits)
    # 使用 Gumbel-Softmax 技术生成一个接近 one-hot 的“软向量”（soft_one_hot）
    soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=False)

    # self.embed.weight 可以理解为（b为batch，n为codebook_size，d为embedding_dim）quantized可以理解n个d维度的latent action primitives进行加权后得出的综合action
    quantized = torch.einsum('b n, n d -> b d', soft_one_hot, self.embed.weight)

    # + kl divergence to the prior loss
    qy = F.softmax(logits, dim=1)
    # 加入熵损失，让编码器“更均匀地”使用所有的 code。
    # 这样就可以用它来鼓励分布 q 接近一个均匀分布，从而防止只用很少的 codewords，提升 codebook 的使用率。
    entropy_loss = 5e-4 * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

    self.temperature = self.linear_decay() 
    return quantized, torch.max(soft_one_hot, dim=-1), entropy_loss
```

## 问题3：模型的head如何定义，输入输出分别对应什么？

回答： 在UniAct模型中，将具身机器人相关的任务head称为`interpreters`，

```
DATASETS_NAME_TO_INTERPRETER = {
    'bridge_dataset': 'MLP_1RGB_7DoFs_4FutureAction',
    'libero-1-rgb': 'MLP_1RGB_7DoFs_4FutureAction',
    ## Add decoder settings for new embodiments!
}

# initialize embodiment-specific low-level interpreters
self.interpreter = nn.ModuleDict()
for domain_name, interpreter in DATASETS_NAME_TO_INTERPRETER.items(): 
    self.interpreter[domain_name] = create_model(interpreter)


    interpreter = self.interpreter[str(domain_name)]
    pred = interpreter(vision_embedding=self.get_vision_embedding(images), 
                        universal_action=universal_action, 
                        proprios = proprios)
    
    action_loss =  (self.loss(pred, action) * action_mask).sum() / action_mask.sum()

# MLP_Decoder的前向过程
class MLP_Decoder(nn.Module):
    def __init__(self,
                universal_action_dim = 128,
                hidden_dim = 512,
                action_dim = 7,
                action_chunking_length = 4):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunking_length = action_chunking_length
        self.head = Mlp(in_features=hidden_dim + universal_action_dim, 
                        hidden_features=action_dim * action_chunking_length * 4, 
                        out_features=action_dim * action_chunking_length)


    def forward(self, 
                vision_embedding: torch.Tensor,  # B V N C
                universal_action: torch.Tensor, # B, ua_dim
                **kwargs): # B, prio_dim
        B = vision_embedding.shape[0]
        inputs = torch.mean(torch.flatten(vision_embedding, 1, 2), dim = 1)
        inputs = torch.cat((inputs, universal_action), dim = -1)
        pred = self.head(inputs).view(B, self.action_chunking_length, self.action_dim) # B, action_dim
        return pred

# ACT_Decoder的前向过程
def forward(self, 
            vision_embedding: torch.Tensor,  # B V N C
            universal_action: torch.Tensor, # B, ua_dim
            proprios: torch.Tensor): # B, prio_dim
    B = vision_embedding.shape[0]
    inputs = torch.cat(
        [
            vision_embedding.flatten(start_dim=1, end_dim=2),
            self.ua_proj(universal_action).unsqueeze(1),
            self.proprio_proj(proprios).unsqueeze(1)
        ], dim = 1
    )
    inputs = inputs + self.input_pos_emb
    query = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
    
    output = self.model.forward(inputs, query) # B ac hidden
    output = self.action_head(output) # B ac 14
    return output
```