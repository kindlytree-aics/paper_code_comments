# UniAct Model结构


## 问题1：如何实现通过线性加权latent action primitives得出universal action？

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

## 问题2：模型的head如何定义，输入输出分别对应什么？

回答： 在UniAct模型中，将具身机器人相关的任务head称为`interpreters`，其输入为

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
```