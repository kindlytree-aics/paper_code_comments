# 关于CrossFormer Model实现的几个问题及解答

## 问题1：该开源代码实现中语言指令（language instructions）和图像融合的FiLM代码具体的体现在什么地方？

回答：其中在函数get_model_config(./scripts/configs/pretrain_config.py)里，定义了ResNet26FILM类为图像类的编码器。
该类成员变量use_film设为True，因此采用了film_conditioning_layer实现将条件变量（一般为language instructions生成的语义张量信息，
如通过BERT输出的对应[CLS]位置的embeddings张量，在代码实现中为UniversalSentenceEncoder，实现位置在./data/utils/text_processing.py中，
在pretrain_config.py的定义方式为:ModuleSpec.create(UniversalSentenceEncoder)）

[分词器实现](https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L159)。根据逻辑和RT-1中类似的实现，该融合在Transformer进行注意力机制计算之前，在Transformer模型输入数据时已经做好相关的融合工作，具体解释如下：  
整个模型类为CrossFormerTransformer，模型输入处理的入口为：https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L89

- 关于task分词器的实现
```
# 关于任务分词器的实现
# task_tokenizers=dict(),但是在.\scripts\configs\pretrain_config.py定义为空字典，因此task不作为专门的tokens序列

for name, tok in self.task_tokenizers.items():
    group_name = f"task_{name}"
    # Receive inputs from tokenizer and cast to embedding size
    tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
    if tokenizer_output is None:
        logging.warning(f"Skipping task tokenizer: {group_name}")
        continue

    task_tokens = nn.Dense(
        self.token_embedding_size, name=f"{group_name}_projection"
    )(tokenizer_output.tokens)
    # task_tokens shape is (batch, n_tokens, token_embedding_size)

    # create positional embedding
    task_pos_enc = self._create_positional_embedding(group_name, task_tokens)

    all_prefix_groups.append(
        PrefixGroup(
            tokens=task_tokens,
            pos_enc=task_pos_enc,
            mask=tokenizer_output.mask,
            name=group_name,
            attention_rules=task_attention_rules,
        )
    )
```

- 关于observations分词器的实现
  
```
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L189

for name, tok in self.observation_tokenizers.items():
    group_name = f"obs_{name}"
    # Receive inputs from tokenizer and cast to embedding size
    tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
    if tokenizer_output is None:
        logging.warning(f"Skipping observation tokenizer: {group_name}")
        continue

    obs_tokens = nn.Dense(
        self.token_embedding_size, name=f"{group_name}_projection"
    )(tokenizer_output.tokens)
    # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

    # create positional embedding
    obs_pos_enc = self._create_positional_embedding(group_name, obs_tokens)

    # Update mask to account for which timesteps are padding
    obs_pad_mask = jnp.logical_and(
        timestep_pad_mask[:, :, None], tokenizer_output.mask
    )

    all_timestep_groups.append(
        TimestepGroup(
            tokens=obs_tokens,
            pos_enc=obs_pos_enc,
            mask=obs_pad_mask,
            name=group_name,
            attention_rules=observation_attention_rules,
        )
    )


# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/scripts/configs/pretrain_config.py#L222

observation_tokenizers=dict(
    primary=ModuleSpec.create(
        ImageTokenizer,
        obs_stack_keys=["image_primary"],
        task_stack_keys=["image_primary"],
        task_film_keys=["language_instruction"],
        encoder=encoder,
    ),
    high=ModuleSpec.create(
        ImageTokenizer,
        obs_stack_keys=["image_high"],
        task_stack_keys=["image_high"],
        task_film_keys=["language_instruction"],
        encoder=encoder,
    ),
    nav=ModuleSpec.create(
        ImageTokenizer,
        obs_stack_keys=["image_nav"],
        task_stack_keys=["image_nav"],
        task_film_keys=[],
        encoder=ModuleSpec.create(ResNet26),
    ),
    left=ModuleSpec.create(
        ImageTokenizer,
        obs_stack_keys=["image_left_wrist"],
        task_stack_keys=[],
        task_film_keys=["language_instruction"],
        encoder=encoder,
    ),
    right=ModuleSpec.create(
        ImageTokenizer,
        obs_stack_keys=["image_right_wrist"],
        task_stack_keys=[],
        task_film_keys=["language_instruction"],
        encoder=encoder,
    ),
    bimanual=ModuleSpec.create(
        LowdimObsTokenizer,
        obs_keys=["proprio_bimanual"],
        dropout_rate=0.2,
    ),
    quadruped=ModuleSpec.create(
        LowdimObsTokenizer,
        obs_keys=["proprio_quadruped"],
    ),
),

# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/components/tokenizers.py#L143
# 注意在数据准备阶段的process_batch函数里，对基于自然语言的task进行编码，具体调用的process_text函数
encoder_input_kwargs = {}
if self.task_film_keys:
    film_inputs = extract_inputs(self.task_film_keys, tasks)
    film_inputs = film_inputs[:, None].repeat(t, axis=1) #扩展t个时间点
    encoder_input_kwargs.update(
        {"cond_var": jnp.reshape(film_inputs, (b * t, -1))}
    )

# run visual encoder
encoder_def = ModuleSpec.instantiate(self.encoder)()
image_tokens = encoder_def(enc_inputs, **encoder_input_kwargs)
image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/components/vit_encoders.py#L261

else:
    if self.use_film:
        assert cond_var is not None, "Cond var is None, nothing to condition on"
        x = FilmConditioning()(x, cond_var)
```

## 问题2：不同的观测tokens类型（如图像和本体观测数据）如何实现的融合，是通过不同的可学习参数的映射层实现不同modality的观测token的融合处理的吗？

回答：get_model_config函数里有定义，针对不同的观测类型有不同的tokenizer，如图像（和自然指令融合）的tokenizer为ImageTokenizer，双臂机器人，四组机器人等的本体感知观测数据的tokenizer为LowdimObsTokenizer。


```
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/scripts/configs/pretrain_config.py#L221

def get_model_config(transformer_size):
    token_embedding_size, transformer_kwargs = common_transformer_sizes(
        transformer_size
    )

    encoder = ModuleSpec.create(ResNet26FILM)
    return dict(
        observation_tokenizers=dict(
            primary=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_primary"],
                task_stack_keys=["image_primary"],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            high=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_high"],
                task_stack_keys=["image_high"],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            nav=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_nav"],
                task_stack_keys=["image_nav"],
                task_film_keys=[],
                encoder=ModuleSpec.create(ResNet26),
            ),
            left=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_left_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            right=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_right_wrist"],
                task_stack_keys=[],
                task_film_keys=["language_instruction"],
                encoder=encoder,
            ),
            bimanual=ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio_bimanual"],
                dropout_rate=0.2,
            ),
            quadruped=ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio_quadruped"],
            ),
        ),
        task_tokenizers=dict(),
        heads=dict(
            bimanual=ModuleSpec.create(
                L1ActionHead,
                action_horizon=100,
                action_dim=BIMANUAL_ACTION_DIM,
                num_preds=BIMANUAL_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_bimanual",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            single_arm=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=SINGLE_ARM_ACTION_DIM,
                num_preds=SINGLE_ARM_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_single_arm",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            nav=ModuleSpec.create(
                L1ActionHead,
                action_horizon=4,
                action_dim=NAV_ACTION_DIM,
                num_preds=NAV_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_nav",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
            quadruped=ModuleSpec.create(
                L1ActionHead,
                action_horizon=1,
                action_dim=QUADRUPED_ACTION_DIM,
                num_preds=QUADRUPED_ACTION_DIM,
                pool_strategy="pass",
                readout_key="readout_quadruped",
                clip_pred=False,
                loss_weight=1.0,
                constrain_loss_dims=True,
            ),
        ),
        readouts=dict(bimanual=100, single_arm=4, nav=4, quadruped=1),
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
        max_horizon=10,
    )

# 针对不同类型的observations，通过projection映射到embedding_size大小
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L197
for name, tok in self.observation_tokenizers.items():
    group_name = f"obs_{name}"
    # Receive inputs from tokenizer and cast to embedding size
    tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
    if tokenizer_output is None:
        logging.warning(f"Skipping observation tokenizer: {group_name}")
        continue

    obs_tokens = nn.Dense(
        self.token_embedding_size, name=f"{group_name}_projection"
    )(tokenizer_output.tokens)
    # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

    # create positional embedding
    obs_pos_enc = self._create_positional_embedding(group_name, obs_tokens)

    # Update mask to account for which timesteps are padding
    obs_pad_mask = jnp.logical_and(
        timestep_pad_mask[:, :, None], tokenizer_output.mask
    )

    all_timestep_groups.append(
        TimestepGroup(
            tokens=obs_tokens,
            pos_enc=obs_pos_enc,
            mask=obs_pad_mask,
            name=group_name,
            attention_rules=observation_attention_rules,
        )
    )
```


## 问题3：readouts的token具体有哪些，如何定义的，和BERT模型的[CLS]类似吗？

回答：在该开源系统的实现中，readouts不对应任务输入，在实现上以生成位置嵌入的形式存在，但和其他的tokens不是进行elementwise的操作，而是concatenate到observations的tokens后面。

```
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L249
for readout_name in readouts:
    group_name = f"readout_{readout_name}"
    # Readouts do not correspond to any inputs, just positional embeddings
    n_tokens_for_readout = self.readouts[readout_name]
    readout_tokens = jnp.zeros(
        (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size)
    )

    # create positional embedding
    readout_pos_enc = self._create_positional_embedding(
        group_name, readout_tokens
    )
    readout_mask = jnp.ones((batch_size, horizon, n_tokens_for_readout))
    readout_attention_rules = {
        "task_*": AttentionRule.CAUSAL,
        "obs_*": AttentionRule.CAUSAL,
        group_name: AttentionRule.CAUSAL,
    }  # Attend to tasks, all previous observations, and *only it's own own readout*

    all_timestep_groups.append(
        TimestepGroup(
            tokens=readout_tokens,
            pos_enc=readout_pos_enc,
            mask=readout_mask,
            name=group_name,
            attention_rules=readout_attention_rules,
        )
    )

# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L315
def _create_positional_embedding(self, name: str, tokens: jax.Array):
    if tokens.ndim == 3:  # for prefixes
        shape = (1, *tokens.shape[-2:])
    elif (
        tokens.ndim == 4
    ):  # for timesteps, create embedding for max_horizon, then truncate
        shape = (1, self.max_horizon, *tokens.shape[-2:])
    else:
        raise ValueError(f"Invalid tokens shape: {tokens.shape}")

    embedding = self.param(
        f"{name}_pos_embedding",
        nn.initializers.normal(stddev=0.02),
        shape,
    )
    if tokens.ndim == 4:
        # Use only the timesteps we receive as input
        embedding = embedding[:, : tokens.shape[1]]
    return jnp.broadcast_to(embedding, tokens.shape)

```

## 问题4： 异构的机器人数据集如何在一起训练，不同horizone窗口的数据对应的tokens输入序列的长度大小不同，是通过padding和mask相结合进行统一处理的吗？以及有什么标记来将transformer的output的embeddings输出给对应的任务head吗？

回答：首先针对不同的数据集的特定格式定义了转换函数将数据转换成统一的形式：OXE_STANDARDIZATION_TRANSFORMS。
和一般的LLM模型的批处理方法类似，确实是通过mask实现了不同长度序列的数据的统一训练，可以在`generate_attention_mask`函数中看一下具体的实现


```
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L249
# 定义了readout的group具体的name

for readout_name in readouts:
    group_name = f"readout_{readout_name}"
    # Readouts do not correspond to any inputs, just positional embeddings
    n_tokens_for_readout = self.readouts[readout_name]
    readout_tokens = jnp.zeros(
        (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size)
    )

    # create positional embedding
    readout_pos_enc = self._create_positional_embedding(
        group_name, readout_tokens
    )
    readout_mask = jnp.ones((batch_size, horizon, n_tokens_for_readout))
    readout_attention_rules = {
        "task_*": AttentionRule.CAUSAL,
        "obs_*": AttentionRule.CAUSAL,
        group_name: AttentionRule.CAUSAL,
    }  # Attend to tasks, all previous observations, and *only it's own own readout*

    all_timestep_groups.append(
        TimestepGroup(
            tokens=readout_tokens,
            pos_enc=readout_pos_enc,
            mask=readout_mask,
            name=group_name,
            attention_rules=readout_attention_rules,
        )
    )

# 并在此位置对输出按groun_name生成outputs字典
# 模型输出的单一序列拆解回原始组结构
https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L292
outputs.update(
    {
        group.name: TokenGroup(group.tokens, group.mask)
        for group in timestep_outputs
    }
)

# 进一步输送给多个head
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/crossformer_module.py#L366

head_outputs = {}
for head_name, head in self.heads.items():
    head_outputs[head_name] = head(transformer_outputs, train=train)
return transformer_outputs, head_outputs


# 每一个head根据对应的tokens输出进行推理
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/components/action_heads.py#L125

token_group = transformer_outputs[self.readout_key]
assert token_group.tokens.ndim == 4, (
    f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
    f"but got shape {token_group.tokens.shape}"
)
if self.pool_strategy == "use_map":  # Multi-head attention pooling
    embeddings = self.map_head(token_group, train=train)[:, :, 0]
elif self.pool_strategy == "mean":  # mean pooling
    embeddings = token_group.tokens.mean(axis=-2)
elif self.pool_strategy == "pass":
    embeddings = token_group.tokens
else:
    raise ValueError(f"{self.pool_strategy} not implemented!")

if len(embeddings.shape) == 3:
    # Implies embeddings is (batch_size, window_size, embedding_size)
    mean = self.mean_proj(embeddings)
    mean = rearrange(
        mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
    )
else:
    # Assumes embeddings is (batch_size, window_size, H, embedding_size)
    assert embeddings.shape[-2] == self.action_horizon
    mean = self.mean_proj(embeddings)

if self.clip_pred:
    mean = jnp.tanh(mean / self.max_action) * self.max_action

return mean


# 训练时还需要计算loss
https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/model/components/action_heads.py#L161

if self.constrain_loss_dims:
    # when using separate heads we can constrain the loss to the action dimensions and action horizon specific to this head
    actions = actions[:, :, : self.action_horizon, : self.action_dim]
    action_pad_mask = action_pad_mask[
        :, :, : self.action_horizon, : self.action_dim
    ]

# (batch, window_size, action_horizon, action_dim)
mean = self(transformer_outputs, train=train)

if action_head_mask is None:
    action_head_mask = jnp.ones(mean.shape[0], dtype=bool)

# combine the timestep pad mask with the action pad mask and the action head mask
mask = (
    timestep_pad_mask[:, :, None, None]
    & action_pad_mask
    & action_head_mask[:, None, None, None]
)

loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
return loss, metrics
```


## 问题5： 异构的机器人数据集处理方面有哪些特殊的需要注意的地方？

回答：回答：首先针对不同的数据集的特定格式定义了转换函数将数据转换成统一的形式：OXE_STANDARDIZATION_TRANSFORMS


```
# https://github.com/rail-berkeley/crossformer/blob/4a56b64411e7ec039ea6cce6bbbe65a38f677db1/crossformer/data/oxe/oxe_standardization_transforms.py#L1070

OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_dataset_transform,
    "taco_extra": taco_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "omnimimic_gnm_dataset": omnimimic_gnm_transform,
    "aloha_dagger_dataset": aloha_dataset_transform,
    "aloha_mobile_dataset": aloha_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    "mujoco_manip": mujoco_manip_dataset_transform,
    "go1": go1_dataset_transform,
    "go1_real_dataset": go1_real_dataset_transform,
    "a1": go1_dataset_transform,
    "aloha_pen_uncap_diverse_dataset": aloha_pen_uncap_dataset_transform,
    "aloha_new_sushi_dataset": aloha_pen_uncap_dataset_transform,
    "aloha_dough_cut_dataset": aloha_dough_dataset_transform,
    "aloha_lucy_dataset": aloha_dough_dataset_transform,
    "aloha_drawer_dataset": aloha_dough_dataset_transform,
    "aloha_pick_place_dataset": aloha_dough_dataset_transform,
    "aloha_static_dataset": aloha_dough_dataset_transform,
    "aloha_sushi_cut_full_dataset": aloha_dough_dataset_transform,
    "droid": droid_dataset_transform,
    "droid_wipe": droid_dataset_transform,
    "droid_flip_pot_upright": droid_dataset_transform,
}
```

## 问题：单臂机器人的任务一般是抓取和放置，因此一般的action只需要开始和任务完成时的动作，其他的可以交给规划来完成？？

from chatgpt
你说得非常对！对于单臂机器人来说，**任务（如抓取和放置）本质上可以被简化为几个关键帧或关键动作**，其余的动作可以由底层控制或规划器来平滑补全。这种做法在许多工业和研究系统中都非常常见。

---

## ✅ 为什么单臂机器人任务通常只需要少量动作（例如开始和结束）？

### 1. **任务结构简单**
- 抓取-放置任务往往分为：
  - **接近目标（approach）**
  - **闭合夹爪（grasp）**
  - **移动到目标位置**
  - **释放（release）**
- 中间的过程，比如路径规划、避障等，完全可以通过**运动规划器**（如 MoveIt、RRT、CHOMP）或**轨迹生成器**自动完成。

---

### 2. **关键帧规划思路**
你可以把单臂动作看成是：
```python
关键动作序列 = [
  {"pose": grasp_pose, "action": "close_gripper"},
  {"pose": place_pose, "action": "open_gripper"}
]
```

模型只需预测这两个 pose，规划器负责在这两个关键帧之间插值出平滑的轨迹。

---

### 3. **预测 horizon 小可以减少负担**
- 在 imitation learning 或 transformer-based policy 中，设置 `action_horizon=4` 足以覆盖“关键帧+少量过渡动作”。
- 太长的 action horizon 对单臂任务反而会引入**不必要的复杂性**，且预测不稳定。

---

## 🚧 举个反例：为什么 bimanual 不行？

- 双臂任务中，两个手臂要**强耦合协同**，例如左手扶稳、右手拧瓶盖。
- 中间过程无法简单插值，必须同时考虑两个手的时序和空间配合，不能简单交给 planner。
- 所以需要长的 `action_horizon` 来捕捉复杂协同策略。

---

## ✅ 总结

| 任务类型       | 控制策略                    | 模型预测需求       |
|----------------|-----------------------------|--------------------|
| 单臂抓放       | 关键帧 + 运动规划           | 预测少量动作       |
| 双臂协同       | 整体轨迹规划（时序耦合）     | 预测长序列         |
| 四足步态控制   | 实时反馈控制                 | 每步实时预测       |
