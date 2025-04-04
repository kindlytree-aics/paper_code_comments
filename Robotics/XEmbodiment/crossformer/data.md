# Dataset


## 问题0：代码实现中数据集处理的代码流程关键路径有哪些？
- `make_interleaved_dataset`
```
# train.py中有如下的代码实现


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
```
- 2
- 3

## 问题1：除了自己制作的数据集，开源的数据集的使用方式是提前下载，并使用一些开源的数据集的处理的代码进行预处理吗？
- openx-embodiment数据集如何下载，以及如何处理？
- 
- 


## 问题2： 不同的数据集最后处理成什么样的统一格式？是observations，task等为key的字典吗？具体的转换操作如何形成


## 问题3：数据集的配比是怎样的，以及如何实现的？