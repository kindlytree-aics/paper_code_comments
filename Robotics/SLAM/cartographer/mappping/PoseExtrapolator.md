# 姿态外推器原理分析

## 刚体几何变换基础

```
.\cartographer\cartographer\transform\rigid_transform.h
//沿 X、Y、Z 三个轴的平移。
Vector translation_; 
//四元数是一种避免万向锁（gimbal lock）问题、稳定且高效的 3D 旋转表示。
//四元数表示的旋转可以转换为旋转矩阵（3x3）或欧拉角。  
Quaternion rotation_;
```

## 姿态外推器的作用

PoseExtrapolator 利用高频（通常几百赫兹）运动传感器（IMU/Odom）数据，在低频扫描数据（Lidar 或相机等用于扫描匹配的传感器频率较低，通常几十赫兹）之间架起了桥梁，既实现了对运动畸变的精确校正，又为关键的扫描匹配优化步骤提供了高质量的初始猜测，是 Cartographer 实现高精度和高鲁棒性局部 SLAM 的关键组件之一。

去畸变（运动补偿的实现分析）

```
std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  //传感器数据时间同步
  //range_data_collator_ 对点云数据进行传感器同步（对多激光头或异步传感器组合有用）
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





