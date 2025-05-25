# 2D Local SLAM算法分析

## 2D Local SLAM的关键实现数据结构和流程分析

关键的数据结构有类`LocalTrajectoryBuilder2D`,`Submap2D`,`PoseExtrapolator`，`ActiveSubmaps2D`等等。

其中类`LocalTrajectoryBuilder2D`起到了类似于集中管理控制器的角色，一些关键的函数的作用简要分析如下：
- 函数`AddRangeData`实现了lidar数据的同步和融合(通过RangeDataCollator类实现)，并实现了点云点级别细粒度的姿态外推器姿态估计和坐标对齐，实现了点云的运动畸变矫正，并对点云点的距离做了限制处理（在一定范围内的点云点算有效返回returns，其他的为丢失misses）。并在累积的点云帧数据达到设定的阈值的时候调用`AddAccumulatedRangeData`函数实现多帧点云数据的聚集
  - 关于点云的畸变矫正：畸变矫正体现在基于姿态外推器的当前的线速度和imu的角度预积分等估计点云点发射时的机器人位姿，将给予此位姿进行变换对齐到统一的坐标系下进行矫正，由于不同点的时间不同，因此机器人位姿也不同，通过每个点的即时位姿实现每个点的位置的坐标对齐。
- 函数`AddAccumulatedRangeData`的主要作用体现在几个方面：
  - 聚合的点云数据一并作为ScanMatch位姿优化的数据，从ScanMatch算法计算优化的位姿后；
  - 将优化后更精确的位姿同步到姿势外推器，外推器基于更精确的位姿序列做最近线速度估计，基于imu_tracker之间的积分角度差异实现旋转的估计；
  - 将点云数据插到局部地图中，通过点云信息更新地图的表示信息


具体代码如下，并在关键代码处加了注释。
//将每个激光点的发射原点和击中点都变换到了这个统一的目标坐标系下。实现了运动畸变校正。
```
std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  //传感器数据时间同步
  //range_data_collator_ 对点云数据进行传感器同步（对多激光传感器设备或异步传感器组合有用）
  auto synchronized_data =
      range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);
  if (synchronized_data.ranges.empty()) {
    LOG(INFO) << "Range data collator filling buffer.";
    return nullptr;
  }

  //基准时间戳 (synchronized_data.time)
  const common::Time& time = synchronized_data.time;
  // Initialize extrapolator now if we do not ever use an IMU.
  if (!options_.use_imu_data()) {
    InitializeExtrapolator(time);
  }

  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator with our first IMU message, we
    // cannot compute the orientation of the rangefinder.
    LOG(INFO) << "Extrapolator not yet initialized.";
    return nullptr;
  }

  CHECK(!synchronized_data.ranges.empty());
  // TODO(gaschler): Check if this can strictly be 0.
  CHECK_LE(synchronized_data.ranges.back().point_time.time, 0.f);
  //一般time为当前同步帧的截止时间，ranges.front()为第一个点云点
  //一般其point_time.time为相对于time的时间，一般为负值
  const common::Time time_first_point =
      time +
      common::FromSeconds(synchronized_data.ranges.front().point_time.time);
  if (time_first_point < extrapolator_->GetLastPoseTime()) {
    LOG(INFO) << "Extrapolator is still initializing.";
    return nullptr;
  }

  std::vector<transform::Rigid3f> range_data_poses;
  range_data_poses.reserve(synchronized_data.ranges.size());
  bool warned = false;
  for (const auto& range : synchronized_data.ranges) {
    //为每个点计算精确的时间戳
    //它内部存储了一个相对于基准时间戳 time 的时间偏移量 range.point_time.time。
    //通过将基准时间戳与这个偏移量相加，计算出该点被传感器实际测量到的精确绝对时间 time_point。
    common::Time time_point = time + common::FromSeconds(range.point_time.time);
    if (time_point < extrapolator_->GetLastExtrapolatedTime()) {
      if (!warned) {
        LOG(ERROR)
            << "Timestamp of individual range data point jumps backwards from "
            << extrapolator_->GetLastExtrapolatedTime() << " to " << time_point;
        warned = true;
      }
      time_point = extrapolator_->GetLastExtrapolatedTime();
    }
    //利用姿态外推器 extrapolator_，根据上一步计算出的每个点的精确时间戳 time_point，
    //查询（外推出）传感器在该精确时刻的位姿（位置和姿态）。
    //这个外推是基于 IMU 和/或 Odometry 数据积分得到的。
    //这些计算出的、对应每个点测量时刻的位姿被存储在 range_data_poses 向量中。
    range_data_poses.push_back(
        extrapolator_->ExtrapolatePose(time_point).cast<float>());
  }

  if (num_accumulated_ == 0) {
    // 'accumulated_range_data_.origin' is uninitialized until the last
    // accumulation.
    accumulated_range_data_ = sensor::RangeData{{}, {}, {}};
  }

  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses.
  for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
    const sensor::TimedRangefinderPoint& hit =
        synchronized_data.ranges[i].point_time;
    //对于第 i 个点 hit（它原始的坐标是在该点测量时刻的传感器坐标系下），
    //使用步骤 3 中计算出的、该点对应时刻的位姿 range_data_poses[i] 将其变换到局部 SLAM 坐标系 (local frame) 下
    const Eigen::Vector3f origin_in_local =
        range_data_poses[i] *
        synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);
    sensor::RangefinderPoint hit_in_local =
        range_data_poses[i] * sensor::ToRangefinderPoint(hit);
    const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
    const float range = delta.norm();
    //将这些已经校正过畸变的点 (hit_in_local) 添加到累积点云数据 (accumulated_range_data_) 中，用于后续的滤波、重力对齐和扫描匹配。
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit_in_local);
      } else {
        hit_in_local.position =
            origin_in_local +
            options_.missing_data_ray_length() / range * delta;
        accumulated_range_data_.misses.push_back(hit_in_local);
      }
    }
  }
  ++num_accumulated_;

  if (num_accumulated_ >= options_.num_accumulated_range_data()) {
    const common::Time current_sensor_time = synchronized_data.time;
    absl::optional<common::Duration> sensor_duration;
    if (last_sensor_time_.has_value()) {
      sensor_duration = current_sensor_time - last_sensor_time_.value();
    }
    last_sensor_time_ = current_sensor_time;
    num_accumulated_ = 0;
    //使用外推器估计当前时刻的重力方向
    const transform::Rigid3d gravity_alignment = transform::Rigid3d::Rotation(
        extrapolator_->EstimateGravityOrientation(time));
    // TODO(gaschler): This assumes that 'range_data_poses.back()' is at time
    // 'time'.
    accumulated_range_data_.origin = range_data_poses.back().translation();
    //将累积点云转换到重力对齐坐标系
    //调用 AddAccumulatedRangeData() 进行实际的匹配和建图
    //最后返回一个 MatchingResult
    return AddAccumulatedRangeData(
        time,
        TransformToGravityAlignedFrameAndFilter(
            gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
            accumulated_range_data_),
        gravity_alignment, sensor_duration);
  }
  return nullptr;
}
```


```
std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddAccumulatedRangeData(
    const common::Time time,
    const sensor::RangeData& gravity_aligned_range_data,
    const transform::Rigid3d& gravity_alignment,
    const absl::optional<common::Duration>& sensor_duration) {
  if (gravity_aligned_range_data.returns.empty()) {
    LOG(WARNING) << "Dropped empty horizontal range data.";
    return nullptr;
  }

  // Computes a gravity aligned pose prediction.
  const transform::Rigid3d non_gravity_aligned_pose_prediction =
      extrapolator_->ExtrapolatePose(time);
  const transform::Rigid2d pose_prediction = transform::Project2D(
      non_gravity_aligned_pose_prediction * gravity_alignment.inverse());

  const sensor::PointCloud& filtered_gravity_aligned_point_cloud =
      sensor::AdaptiveVoxelFilter(gravity_aligned_range_data.returns,
                                  options_.adaptive_voxel_filter_options());
  if (filtered_gravity_aligned_point_cloud.empty()) {
    return nullptr;
  }

  // local map frame <- gravity-aligned frame
  std::unique_ptr<transform::Rigid2d> pose_estimate_2d =
      ScanMatch(time, pose_prediction, filtered_gravity_aligned_point_cloud);
  if (pose_estimate_2d == nullptr) {
    LOG(WARNING) << "Scan matching failed.";
    return nullptr;
  }
  const transform::Rigid3d pose_estimate =
      transform::Embed3D(*pose_estimate_2d) * gravity_alignment;
  extrapolator_->AddPose(time, pose_estimate);

  sensor::RangeData range_data_in_local =
      TransformRangeData(gravity_aligned_range_data,
                         transform::Embed3D(pose_estimate_2d->cast<float>()));
  std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
      time, range_data_in_local, filtered_gravity_aligned_point_cloud,
      pose_estimate, gravity_alignment.rotation());

  const auto wall_time = std::chrono::steady_clock::now();
  if (last_wall_time_.has_value()) {
    const auto wall_time_duration = wall_time - last_wall_time_.value();
    kLocalSlamLatencyMetric->Set(common::ToSeconds(wall_time_duration));
    if (sensor_duration.has_value()) {
      kLocalSlamRealTimeRatio->Set(common::ToSeconds(sensor_duration.value()) /
                                   common::ToSeconds(wall_time_duration));
    }
  }
  const double thread_cpu_time_seconds = common::GetThreadCpuTimeSeconds();
  if (last_thread_cpu_time_seconds_.has_value()) {
    const double thread_cpu_duration_seconds =
        thread_cpu_time_seconds - last_thread_cpu_time_seconds_.value();
    if (sensor_duration.has_value()) {
      kLocalSlamCpuRealTimeRatio->Set(
          common::ToSeconds(sensor_duration.value()) /
          thread_cpu_duration_seconds);
    }
  }
  last_wall_time_ = wall_time;
  last_thread_cpu_time_seconds_ = thread_cpu_time_seconds;
  return absl::make_unique<MatchingResult>(
      MatchingResult{time, pose_estimate, std::move(range_data_in_local),
                     std::move(insertion_result)});
}
```

## 2D Local SLAM的子地图的维护和更新规则

子地图的创建时机： 当机器人移动了一段距离，或者收集了足够多的激光雷达数据（由 TRAJECTORY_BUILDER_2D.submaps.num_range_data 等参数控制）后，LocalTrajectoryBuilder2D 会决定结束当前的子地图（如果存在且活动）并创建一个新的子地图。如下代码严格定义了子地图集合中子地图的创建和结束的规则。

```
std::vector<std::shared_ptr<const Submap2D>> ActiveSubmaps2D::InsertRangeData(
    const sensor::RangeData& range_data) {
  if (submaps_.empty() ||
      submaps_.back()->num_range_data() == options_.num_range_data()) {
    AddSubmap(range_data.origin.head<2>());
  }
  for (auto& submap : submaps_) {
    submap->InsertRangeData(range_data, range_data_inserter_.get());
  }
  if (submaps_.front()->num_range_data() == 2 * options_.num_range_data()) {
    submaps_.front()->Finish();
  }
  return submaps();
}

void ActiveSubmaps2D::AddSubmap(const Eigen::Vector2f& origin) {
  if (submaps_.size() >= 2) {
    // This will crop the finished Submap before inserting a new Submap to
    // reduce peak memory usage a bit.
    CHECK(submaps_.front()->insertion_finished());
    submaps_.erase(submaps_.begin());
  }
  submaps_.push_back(absl::make_unique<Submap2D>(
      origin,
      std::unique_ptr<Grid2D>(
          static_cast<Grid2D*>(CreateGrid(origin).release())),
      &conversion_tables_));
}

```

//对上面代码的解释： interpretation from google  Gemini 2.5 Pro Preview 05-06
submaps_ 容器的状态：
最多两个子地图： 一个是当前正在主要构建的 (submaps_.back())，另一个是前一个（submaps_.front()，如果存在）。
submaps_.back() (最新的)： 新的 RangeData 会主要贡献给这个子地图。当它累积了 options_.num_range_data() 个扫描后，下一次 InsertRangeData 调用会触发 AddSubmap。
submaps_.front() (较旧的)：
在新的子地图被创建后，它成为 submaps_.front()。
它仍然会通过 for 循环接收新的 RangeData，直到它累积了 2 * options_.num_range_data() 个扫描。
当它达到 2 * options_.num_range_data() 时，它被 Finish()。
当再下一个新的子地图被创建时（即 submaps_.back() 又满了），这个已经 Finish() 的 submaps_.front() 会被 AddSubmap 中的 submaps_.erase(submaps_.begin()) 移除。
RangeData 的插入行为：
一份 RangeData 确实会同时插入到 submaps_.back() 和 submaps_.front() (如果存在) 中。
这意味着，在从一个子地图过渡到下一个子地图的期间，会有 options_.num_range_data() 数量的扫描数据被同时插入到两个相邻的子地图中。
例如，假设 options_.num_range_data() 是 100。
Submap A 开始构建，接收扫描 1-100。
扫描 100 到达后，Submap B 被创建。
扫描 101-200 会被同时插入到 Submap A 和 Submap B。
当 Submap A 接收到第 200 个扫描（总共 2 * options_.num_range_data()）时，它被 Finish()。
当 Submap B 接收到它的第 100 个扫描（即全局扫描的第 200 个）时，Submap C 被创建，同时 Submap A 被从 submaps_ 中移除。
扫描 201-300 会被同时插入到 Submap B 和 Submap C。
这种设计的目的/影响：
平滑过渡和重叠： 允许相邻子地图之间有显著的数据重叠。这对于后续的位姿图优化和闭环检测非常重要，因为它确保了子地图之间的连接性。
扫描匹配基础： 这两个子地图（submaps_.front() 和 submaps_.back()）都是扫描匹配的目标。拥有一个“更成熟”的（submaps_.front()，即使它仍在接收一些数据）和一个“更新的”（submaps_.back()）子地图进行匹配，可以提高鲁棒性。
为什么不只插入一个？ 如果只插入到 submaps_.back()，那么当切换子地图时，新的子地图会从零开始构建。而当前的实现允许新的子地图（submaps_.back()）和前一个子地图（submaps_.front()）共享一段时间的数据，使得它们的初始部分有重叠。这可能有助于更早地稳定新子地图的结构。


### LocalSLAM的大体算法处理流程逻辑
- ScanMatch优化位姿
- 外推器记录优化后位姿并更新速度等估计值
- 将激光点云帧坐标转换后就插入到当前活动子图中

```
// local map frame <- gravity-aligned frame
std::unique_ptr<transform::Rigid2d> pose_estimate_2d =
    ScanMatch(time, pose_prediction, filtered_gravity_aligned_point_cloud);
if (pose_estimate_2d == nullptr) {
  LOG(WARNING) << "Scan matching failed.";
  return nullptr;
}
const transform::Rigid3d pose_estimate =
    transform::Embed3D(*pose_estimate_2d) * gravity_alignment;
extrapolator_->AddPose(time, pose_estimate);

sensor::RangeData range_data_in_local =
    TransformRangeData(gravity_aligned_range_data,
                        transform::Embed3D(pose_estimate_2d->cast<float>()));
std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
    time, range_data_in_local, filtered_gravity_aligned_point_cloud,
    pose_estimate, gravity_alignment.rotation());

std::unique_ptr<LocalTrajectoryBuilder2D::InsertionResult>
LocalTrajectoryBuilder2D::InsertIntoSubmap(
    const common::Time time, const sensor::RangeData& range_data_in_local,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud,
    const transform::Rigid3d& pose_estimate,
    const Eigen::Quaterniond& gravity_alignment) {
  if (motion_filter_.IsSimilar(time, pose_estimate)) {
    return nullptr;
  }
  std::vector<std::shared_ptr<const Submap2D>> insertion_submaps =
      active_submaps_.InsertRangeData(range_data_in_local);
  return absl::make_unique<InsertionResult>(InsertionResult{
      std::make_shared<const TrajectoryNode::Data>(TrajectoryNode::Data{
          time,
          gravity_alignment,
          filtered_gravity_aligned_point_cloud,
          {},  // 'high_resolution_point_cloud' is only used in 3D.
          {},  // 'low_resolution_point_cloud' is only used in 3D.
          {},  // 'rotational_scan_matcher_histogram' is only used in 3D.
          pose_estimate}),
      std::move(insertion_submaps)});
}

void Submap2D::InsertRangeData(
    const sensor::RangeData& range_data,
    const RangeDataInserterInterface* range_data_inserter) {
  CHECK(grid_);
  CHECK(!insertion_finished());
  range_data_inserter->Insert(range_data, grid_.get());
  set_num_range_data(num_range_data() + 1);
}

void ProbabilityGridRangeDataInserter2D::Insert(
    const sensor::RangeData& range_data, GridInterface* const grid) const {
  ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);
  CHECK(probability_grid != nullptr);
  // By not finishing the update after hits are inserted, we give hits priority
  // (i.e. no hits will be ignored because of a miss in the same cell).
  CastRays(range_data, hit_table_, miss_table_, options_.insert_free_space(),
           probability_grid);
  probability_grid->FinishUpdate();
}
```

### 地图建图过程以及坐标系定义及点云点坐标变换方法
//当一个新的子地图被创建时，插入到这个新子地图的第一个激光扫描数据（或者说，机器人获取这个扫描数据时的位姿）定义了这个新子地图的局部坐标系原点。
//这个 local_pose_存储的是从该子地图的局部坐标系到全局 map 坐标系的变换。local_pose_ 包含了平移 (translation) 和旋转 (rotation)。

```
Submap2D::Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
                   ValueConversionTables* conversion_tables)
    : Submap(transform::Rigid3d::Translation(
          Eigen::Vector3d(origin.x(), origin.y(), 0.))),
      conversion_tables_(conversion_tables) {
  grid_ = std::move(grid);
}

Submap(const transform::Rigid3d& local_submap_pose)
    : local_pose_(local_submap_pose) {}

static Rigid3 Translation(const Vector& vector) {
  return Rigid3(vector, Quaternion::Identity());
}

void ProbabilityGridRangeDataInserter2D::Insert(
    const sensor::RangeData& range_data, GridInterface* const grid) const {
  ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);
  CHECK(probability_grid != nullptr);
  // By not finishing the update after hits are inserted, we give hits priority
  // (i.e. no hits will be ignored because of a miss in the same cell).
  CastRays(range_data, hit_table_, miss_table_, options_.insert_free_space(),
           probability_grid);
  probability_grid->FinishUpdate();
}
```

#### 坐标对齐变换的过程分析
#####  一开始点云左边变换到机器人本体坐标系下，在ros节点处进行
```
void SensorBridge::HandleRangefinder(
    const std::string& sensor_id, const carto::common::Time time,
    const std::string& frame_id, const carto::sensor::TimedPointCloud& ranges) {
  if (!ranges.empty()) {
    CHECK_LE(ranges.back().time, 0.f);
  }
  const auto sensor_to_tracking =
      tf_bridge_.LookupToTracking(time, CheckNoLeadingSlash(frame_id));
  if (sensor_to_tracking != nullptr) {
    if (IgnoreMessage(sensor_id, time)) {
      LOG(WARNING) << "Ignored Rangefinder message from sensor " << sensor_id
                   << " because sensor time " << time
                   << " is not before last Rangefinder message time "
                   << latest_sensor_time_[sensor_id];
      return;
    }
    latest_sensor_time_[sensor_id] = time;
    trajectory_builder_->AddSensorData(
        sensor_id, carto::sensor::TimedPointCloudData{
                       time, sensor_to_tracking->translation().cast<float>(),
                       carto::sensor::TransformTimedPointCloud(
                           ranges, sensor_to_tracking->cast<float>())});
  }
}

```

##### 每一个点云点的时间的位姿估计，对点云点进行去畸变矫正处理

当一个低频的传感器数据（如激光扫描）到达时，外推器可以提供在该时刻的机器人较为精确的位姿估计。
这个位姿是相对于SLAM过程开始时定义的“地图”坐标系（或称“世界”坐标系）的。外推器内部维护的位姿状态是基于这个全局参考系的。

```
std::vector<transform::Rigid3f> range_data_poses;
range_data_poses.reserve(synchronized_data.ranges.size());
bool warned = false;
for (const auto& range : synchronized_data.ranges) {
  common::Time time_point = time + common::FromSeconds(range.point_time.time);
  if (time_point < extrapolator_->GetLastExtrapolatedTime()) {
    if (!warned) {
      LOG(ERROR)
          << "Timestamp of individual range data point jumps backwards from "
          << extrapolator_->GetLastExtrapolatedTime() << " to " << time_point;
      warned = true;
    }
    time_point = extrapolator_->GetLastExtrapolatedTime();
  }
  range_data_poses.push_back(
      extrapolator_->ExtrapolatePose(time_point).cast<float>());
}

if (num_accumulated_ == 0) {
  // 'accumulated_range_data_.origin' is uninitialized until the last
  // accumulation.
  accumulated_range_data_ = sensor::RangeData{{}, {}, {}};
}
```

##### 进行畸变矫正对齐到统一的地图坐标系下`hit_in_local`，这段代码将点云点坐标变换到地图坐标系下。


`origins`通常代表传感器在其自身坐标系中的原点（不是机器人此时相对于全局坐标系下的位置），或者传感器在其安装的机器人基座 (base_link/tracking_frame) 坐标系中的固定位置
`origin_in_local`将传感器的原点变换到局部坐标下，同理`hit_in_local`将点云点坐标变换到局部坐标下。


```
  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses.
  for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
    const sensor::TimedRangefinderPoint& hit =
        synchronized_data.ranges[i].point_time;
    const Eigen::Vector3f origin_in_local =
        range_data_poses[i] *
        synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);
    sensor::RangefinderPoint hit_in_local =
        range_data_poses[i] * sensor::ToRangefinderPoint(hit);
    const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
    const float range = delta.norm();
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit_in_local);
      } else {
        hit_in_local.position =
            origin_in_local +
            options_.missing_data_ray_length() / range * delta;
        accumulated_range_data_.misses.push_back(hit_in_local);
      }
    }
  }
```

##### 后面将其反变换到了局部坐标系下，代码如下

//在累积完成后，调用 TransformToGravityAlignedFrameAndFilter，这个函数接收地图坐标系下的点，并通过一个变换（涉及到机器人位姿的逆和重力对齐旋转）将它们转换到某个中间的、可能是机器人本地的、重力对齐的坐标系进行滤波。
//AddAccumulatedRangeData 函数会接收这个滤波后的点云（在那个中间坐标系下）以及其他信息，然后进行扫描匹配和最终的子图插入。在这些后续步骤中，才需要将点云变换到目标子图的局部坐标系。
```
accumulated_range_data_.origin = range_data_poses.back().translation();
return AddAccumulatedRangeData(
    time,
    TransformToGravityAlignedFrameAndFilter(
        gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
        accumulated_range_data_),
    gravity_alignment, sensor_duration);
```
- 然后又将其变换到全局坐标系下，并插入到子地图
```
sensor::RangeData range_data_in_local =
  TransformRangeData(gravity_aligned_range_data,
                      transform::Embed3D(pose_estimate_2d->cast<float>()));
std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
  time, range_data_in_local, filtered_gravity_aligned_point_cloud,
  pose_estimate, gravity_alignment.rotation());

```

- 点云点插入到地图更新地图表示的逻辑实现
```
void ProbabilityGridRangeDataInserter2D::Insert(
    const sensor::RangeData& range_data, GridInterface* const grid) const {
  ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);
  CHECK(probability_grid != nullptr);
  // By not finishing the update after hits are inserted, we give hits priority
  // (i.e. no hits will be ignored because of a miss in the same cell).
  CastRays(range_data, hit_table_, miss_table_, options_.insert_free_space(),
           probability_grid);
  probability_grid->FinishUpdate();
}

void GrowAsNeeded(const sensor::RangeData& range_data,
                  ProbabilityGrid* const probability_grid) {
  //Aligned"（对齐的）：意味着包围盒的边与坐标轴平行。它不是一个任意旋转的矩形，它的边总是平行于 X 轴和 Y 轴（对于2D情况）或 X, Y, Z 轴（对于3D情况）。
  //当使用单个点来初始化一个 AlignedBox 时，这个点同时定义了包围盒的最小值点 (min()) 和最大值点 (max())。
  //几何上，这意味着创建的包围盒是一个退化（degenerate）的矩形，它的宽度和高度都为零。它仅仅包含了构造时传入的那一个点。
  //bounding_box.min() 将等于 range_data.origin.head<2>()。
  //bounding_box.max() 也将等于 range_data.origin.head<2>()。
  Eigen::AlignedBox2f bounding_box(range_data.origin.head<2>());
  // Padding around bounding box to avoid numerical issues at cell boundaries.
  constexpr float kPadding = 1e-6f;
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    bounding_box.extend(hit.position.head<2>());
  }
  for (const sensor::RangefinderPoint& miss : range_data.misses) {
    bounding_box.extend(miss.position.head<2>());
  }
  probability_grid->GrowLimits(bounding_box.min() -
                               kPadding * Eigen::Vector2f::Ones());
  probability_grid->GrowLimits(bounding_box.max() +
                               kPadding * Eigen::Vector2f::Ones());
}
```

### 姿态外推器、ImuTracker以及LocalTrajectoryBuilder之间的位姿估计于优化的逻辑关系
- 姿态外推器
- LocalTrajectoryBuilder
- ImuTracker 

### 上述代码实现中，将位姿转换到和imu重力方向一致的坐标系的作用以及具体方法是什么？

为什么需要这一系列操作？
2D SLAM的假设: 传统的2D SLAM（如Cartographer的2D模式）通常假设机器人在一个平面上运动，或者至少其姿态变化主要体现在XY平移和绕Z轴的旋转（yaw）。
现实情况: 机器人，尤其是在不平坦地面上运动的机器人，会产生roll和pitch角。如果直接将带有roll/pitch的3D位姿投影到2D，或者直接使用未对齐的2D激光数据，会导致地图构建错误和定位失败。例如，一个倾斜的激光扫描在投影到2D后会发生畸变。
解决方案:
对齐传感器数据: 利用IMU提供的重力方向，将激光点云旋转到一个虚拟的“水平”平面（即 tracking_frame_ga 的XY平面）。
对齐位姿估计: 相应地，机器人的位姿估计也必须转换到这个“水平”参考系下。这就是 non_gravity_aligned_pose_prediction * gravity_alignment.inverse() 然后 Project2D 所做的事情。
一致性: 这样做保证了用于扫描匹配的点云和机器人位姿都在同一个一致的、重力对齐的2D坐标系下进行表示和比较，从而使得2D扫描匹配算法能够有效工作。

## ScanMatch的实现原理

```
std::unique_ptr<transform::Rigid2d> LocalTrajectoryBuilder2D::ScanMatch(
    const common::Time time, const transform::Rigid2d& pose_prediction,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud) {
  if (active_submaps_.submaps().empty()) {
    return absl::make_unique<transform::Rigid2d>(pose_prediction);
  }
  std::shared_ptr<const Submap2D> matching_submap =
      active_submaps_.submaps().front();
  // The online correlative scan matcher will refine the initial estimate for
  // the Ceres scan matcher.
  transform::Rigid2d initial_ceres_pose = pose_prediction;

  if (options_.use_online_correlative_scan_matching()) {
    const double score = real_time_correlative_scan_matcher_.Match(
        pose_prediction, filtered_gravity_aligned_point_cloud,
        *matching_submap->grid(), &initial_ceres_pose);
    kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
  }

  auto pose_observation = absl::make_unique<transform::Rigid2d>();
  ceres::Solver::Summary summary;
  ceres_scan_matcher_.Match(pose_prediction.translation(), initial_ceres_pose,
                            filtered_gravity_aligned_point_cloud,
                            *matching_submap->grid(), pose_observation.get(),
                            &summary);
  if (pose_observation) {
    kCeresScanMatcherCostMetric->Observe(summary.final_cost);
    const double residual_distance =
        (pose_observation->translation() - pose_prediction.translation())
            .norm();
    kScanMatcherResidualDistanceMetric->Observe(residual_distance);
    const double residual_angle =
        std::abs(pose_observation->rotation().angle() -
                 pose_prediction.rotation().angle());
    kScanMatcherResidualAngleMetric->Observe(residual_angle);
  }
  return pose_observation;
}
```

## 2D SLAM的全局优化过程中，位姿优化的算法PoshGraph2D具体细节原理？


