# Dataset

## RLDS强化学习数据集开源库如何定义数据？以及提供了哪些关键的函数接口，以及接口的使用

## 问题：数据集读取标准化的过程如何实现的？

- `full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)`语句中restructure实现了异构数据的标准化转换，

```
def restructure(traj):
    # apply a standardization function, if provided
    if standardize_fn is not None:
        traj = ModuleSpec.instantiate(standardize_fn)(traj)

```

- 其中 `standardization`为标准化转换函数，针对不同的数据集，其转换函数的实现不一样，针对不同的数据集的转换函数形成了name和函数名对应的字典

```
make_oxe_dataset_kwargs函数里有定义：

dataset_kwargs["standardize_fn"] = ModuleSpec.create(
    OXE_STANDARDIZATION_TRANSFORMS[name]
)

OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
```

## 问题：数据集的按照采样权重进行采样是如何实现的？

- 定义数据集列表及权重

```
datasets = [robot_arm_data, mobile_robot_data, human_demo_data]
sample_weights = [0.5, 0.3, 0.2]  # 机械臂:50%, 移动机器人:30%, 人类示教:20%
# 每次从所有数据集的帧池中，按 sample_weights 的概率随机选择一个帧。例如，若 sample_weights=[0.7, 0.3]，则每个帧有 70% 的概率来自 dataset_a，30% 的概率来自 dataset_b。
```

- 帧级混合与打乱

```
mixed_dataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)
mixed_dataset = mixed_dataset.shuffle(shuffle_buffer_size=5000)
```

- 批量加载

```
train_loader = mixed_dataset.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)
```

## 问题： dl.DLataset.from_rlds(builder, split="all", shuffle=False)该语句使用的dlimp库，其读取后和RLDS的数据格式有区别了对吗？已经没有steps了是吗？（待确认）

- dlimp库的dataset.py文件中 _broadcast_metadata_rlds函数实通过弹出steps并展开其内容到顶层的方式，实现了层级结构的扁平化。

```
def _broadcast_metadata_rlds(i: tf.Tensor, traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    In the RLDS format, each trajectory has some top-level metadata that is explicitly separated out, and a "steps"
    entry. This function moves the "steps" entry to the top level, broadcasting any metadata to the length of the
    trajectory. This function also adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.
    """
    steps = traj.pop("steps")

    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]

    # broadcast metadata to the length of the trajectory
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)

    # put steps back in
    assert "traj_metadata" not in steps
    traj = {**steps, "traj_metadata": metadata}

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    return traj
```

## 问题：代码实现中数据集处理的代码流程关键路径有哪些？

- `make_interleaved_dataset` 为加载数据的主要函数，`interleaved`顾名思义为交叉加载数据集
  - 对于每一个子数据集
    - `make_dataset_from_rlds`实现加载数据集并通过调用 `restructure`实现格式上的一致
    - `apply_trajectory_transforms`实现了对轨迹的转换，如padding，mask等处理，对异构数据保持输入维度一致，以便进行批量训练
    - 加入到整个数据集中
  - `dl.DLataset.sample_from_datasets`函数根据比例进行采样
  - `apply_frame_transforms`对每个样本帧数据进行转换，如对图像进行resize，数据增强等等

```
# train.py中有如下的代码实现

# load datasets
if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
    # create dataset_kwargs_list from oxe_kwargs
    (
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
        FLAGS.config.dataset_kwargs["sample_weights"],
    ) = make_oxe_dataset_kwargs_and_weights(
        **FLAGS.config.dataset_kwargs["oxe_kwargs"]
    )
    del FLAGS.config.dataset_kwargs["oxe_kwargs"]

FLAGS.config.dataset_kwargs.batch_size //= jax.process_count()
train_data = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

train_data_iter = map(
    shard,
    map(
        process_batch,
        train_data.iterator(prefetch=FLAGS.config.prefetch_num_batches),
    ),
)

# .\scripts\configs\pretrain_config.py

window_size = FieldReference(default=5)

dataset_kwargs=get_dataset_config("multi", window_size, 100),

def get_dataset_config(task_cond, window_size, action_horizon):
    traj_transform_kwargs, frame_transform_kwargs = get_augmentation_config(
        task_cond, window_size, action_horizon
    )

    mix = "cross_embodiment"
    assert all(
        [
            any([name in datasets for datasets in HEAD_TO_DATASET.values()])
            for name, weight in OXE_NAMED_MIXES[mix]
        ]
    ), "Dataset in mix doesn't have assigned head."

    return dict(
        oxe_kwargs=dict(
            data_mix=mix,
            data_dir="",
            load_camera_views=("primary", "high", "nav", "left_wrist", "right_wrist"),
            load_proprio=True,
            load_depth=False,
        ),
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        batch_size=512,
        shuffle_buffer_size=50000,
        balance_weights=False,
        traj_transform_threads=48,
        traj_read_threads=48,
    )


# .\crossformer\data\dataset.py

def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overidden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overidden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # default to uniform sampling
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = {}
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        assert (
            dataset_kwargs["name"] not in all_dataset_statistics
        ), f"Duplicate name {dataset_kwargs['name']}"
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        # override global traj transform kwargs with dataset specfic ones
        if "override_traj_transform_kwargs" in dataset_kwargs:
            traj_transform_kwargs.update(
                dataset_kwargs.pop("override_traj_transform_kwargs")
            )

        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    # interleave at the frame level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    dataset = dataset.ignore_errors(log_warning=True)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    dataset.sample_weights = sample_weights
    return dataset

# .\data\oxe\oxe_dataset_mixes.py
```

"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""

OXE_MAGIC_SOUP_BALANCED = [
    ("kuka", 0.14503701874493363),
    ("taco_play", 0.06657998827701668),
    ("taco_extra", 0.015452958868388737),
    ("jaco_play", 0.010914534155076169),
    ("berkeley_cable_routing", 0.005925612796973822),
    ("roboturk", 0.052499238268860826),
    ("nyu_door_opening_surprising_effectiveness", 0.0028565519070650833),
    ("viola", 0.021369612129854),
    ("berkeley_autolab_ur5", 0.027421498380401588),
    ("toto", 0.045595496181288435),
    ("language_table", 0.09863155061985435),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 0.10030032010542056),
    ("austin_buds_dataset_converted_externally_to_rlds", 0.004775432426062442),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.01884652293499813),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.05526993262706029),
    ("austin_sailor_dataset_converted_externally_to_rlds", 0.04943059735717906),
    ("austin_sirius_dataset_converted_externally_to_rlds", 0.03918942829266809),
    ("bc_z", 0.14503701874493363),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 0.00124985520344411),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 0.020472678629801757),
    ("utaustin_mutex", 0.05066099356944051),
    ("berkeley_fanuc_manipulation", 0.017530731149920712),
    ("cmu_stretch", 0.003502058441908362),
    ("droid", 0.001450370187449336),
]

CROSS_EMBODIMENT_TARGET = [
    ("aloha_pen_uncap_diverse_dataset", 0.1),
    ("aloha_new_sushi_dataset", 0.2),
    ("bridge_dataset", 0.2),
    ("a1", 0.1),
    ("droid_wipe", 0.1),
    ("droid_flip_pot_upright", 0.1),
    ("omnimimic_gnm_dataset", 0.2),
]

CROSS_EMBODIMENT = [
    (name, weight * 0.15) for name, weight in OXE_MAGIC_SOUP_BALANCED
] + [(name, weight * 0.85) for name, weight in CROSS_EMBODIMENT_TARGET]

OXE_NAMED_MIXES = {
    "cross_embodiment": CROSS_EMBODIMENT,
}

get_dataset_statistics函数统计action和本体观测数据的数据分布特性，如均值，最大最小，以及轨迹数目等。

make_dataset_from_rlds

问题： RLDS的强化学习数据集有哪些特性，遵循标准的数据格式吗？以及有什么开源的数据集读取加载和处理的代码库可以使用以处理RLDS的数据，机器人方面以RLDS来存储的数据有哪些？

回答： RLDS是google团队提出的开源库，专为强化学习数据库设计，强制要求含有observation，action，reward等核心字段，每一个字段为一个轨迹序列下该字段的完整的list数据

```
# load or compute dataset statistics
if isinstance(dataset_statistics, str):
    with tf.io.gfile.GFile(dataset_statistics, "r") as f:
        dataset_statistics = json.load(f)
elif dataset_statistics is None:
    # 加载全量数据: 使用 dl.DLataset.from_rlds 加载整个数据集（split="all"），不进行打乱。
    full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)

    # 遍历 filter_functions，通过 ModuleSpec.instantiate 实例化过滤函数，并应用到数据集。
    # traj_map(restructure): 对每个轨迹应用 restructure 函数（可能用于调整数据结构）。
    # filter(is_nonzero_length): 过滤掉长度为0的轨迹。
    for filter_fcn_spec in filter_functions:
        full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    if ignore_errors:
        full_dataset = full_dataset.ignore_errors()
    full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)
    # tries to load from cache, otherwise computes on the fly

def restructure(traj):
    # apply a standardization function, if provided
    if standardize_fn is not None:
        traj = ModuleSpec.instantiate(standardize_fn)(traj)

    if not all(k in traj for k in REQUIRED_KEYS):
        raise ValueError(
            f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
            "Did you write a `standardize_fn`?"
        )

    # extracts images, depth images and proprio from the "observation" dict
    traj_len = tf.shape(traj["action"])[0]
    old_obs = traj["observation"]
    new_obs = {}
    for new, old in image_obs_keys.items():
        if old is None:
            new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"image_{new}"] = old_obs[old]

    for new, old in depth_obs_keys.items():
        if old is None:
            new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"depth_{new}"] = old_obs[old]

    if proprio_obs_keys is not None:
        for new, old in proprio_obs_keys.items():
            if old is None:
                new_obs[f"proprio_{new}"] = tf.zeros(
                    (traj_len, proprio_obs_dims[new]), dtype=tf.float32
                )  # padding
            else:
                new_obs[f"proprio_{new}"] = tf.cast(old_obs[old], tf.float32)

    # add timestep info
    new_obs["timestep"] = tf.range(traj_len)

# 并且通过config定义将不同数据集的对应的名称进行统一表示，如fractal20220817_data数据集中将图像数据的image字段对应到primary新字段上

"fractal20220817_data": {
    "image_obs_keys": {
        "primary": "image",
        "high": None,
        "nav": None,
        "left_wrist": None,
        "right_wrist": None,
    },
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "proprio_obs_keys": {"bimanual": None, "quadruped": None},
    "proprio_obs_dims": {"bimanual": 14, "quadruped": 46},
    "proprio_encoding": ProprioEncoding.POS_QUAT,
    "action_encoding": ActionEncoding.EEF_POS,
},
"stanford_robocook_converted_externally_to_rlds": {
    "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
    "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
    "proprio_encoding": ProprioEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
},
"aloha_drawer_dataset": {
    "image_obs_keys": {
        "primary": None,
        "high": "cam_high",
        "nav": None,
        "left_wrist": "cam_left_wrist",
        "right_wrist": "cam_right_wrist",
    },
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "proprio_obs_keys": {"bimanual": "proprio", "quadruped": None},
    "proprio_obs_dims": {"bimanual": 14, "quadruped": 46},
    "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
    "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
},

# OXE_STANDARDIZATION_TRANSFORMS里定义了不同的数据集的个数转化为标注格式的方法，如前面几个：
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_dataset_transform,
    "taco_extra": taco_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    ...

def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory

# 数据集和机器人形态功能（导航，单臂，双臂，四足）的映射关系。
# .\scripts\configs\pretrain_config.py
HEAD_TO_DATASET = {
    "nav": ["omnimimic_gnm_dataset"],
    "single_arm": [
        "bridge_dataset",
        "fractal20220817_data",
        "kuka",
        "taco_play",
        "taco_extra",
        "jaco_play",
        "berkeley_cable_routing",
        "roboturk",
        "nyu_door_opening_surprising_effectiveness",
        "viola",
        "berkeley_autolab_ur5",
        "toto",
        "language_table",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "bc_z",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_stretch",
        "droid",
        "droid_wipe",
        "droid_flip_pot_upright",
    ],
    "bimanual": [
        "aloha_pen_uncap_diverse_dataset",
        "aloha_new_sushi_dataset",
        "aloha_dough_cut_dataset",
        "aloha_lucy_dataset",
        "aloha_drawer_dataset",
        "aloha_pick_place_dataset",
        "aloha_static_dataset",
        "aloha_sushi_cut_full_dataset",
        "aloha_new_sushi_dataset,",
    ],
    "quadruped": ["go1_real_dataset", "a1", "go1"],
}
```

## 对不同形态的机器人数据挑选代表并进行可视化显示

tfds.builder(name, data_dir=data_dir)

- kuka工业机器人： https://www.kuka.cn/zh-cn/products/robotics-systems/industrial-robots
- UR5: https://www.universal-robots.com/
- xArm: https://www.ufactory.cc/xarm-collaborative-robot/ 
- 移动机器人（Mobile Manipulators） https://hello-robot.com/
- https://www.tacbot.cn/ https://github.com/frankaemika/franka_ros

## 问题及任务：单臂机器人最为常见（基于视觉+自然语言指令），ALOHA基于双臂机器人（三个摄像头数据）较为特殊，请对几个较为特殊的机器人数据进行可视化显示


## 在训练时的机器人数据的权重配比以及其配比是如何设置的，以及每个数据集的数据量如何统计（比如多少trajectories，多少frames（observations，actions））

We classify the datasets that are most relevant to each of our evaluation settings
(see Section 4) as the target datasets and up-weight them relative to the other datasets during training.

Our target datasets are BridgeData [36] for WidowX evaluation,ALOHA-multi-task for ALOHA evaluation, GNM [41]
for navigation evaluation, Go1-walk for quadruped evaluation, and Franka-tabletop for Franka evaluation.

## 