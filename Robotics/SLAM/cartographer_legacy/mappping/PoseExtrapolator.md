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

PoseExtrapolator 利用高频（通常几百赫兹）运动传感器（IMU/Odom）数据，在低频扫描数据（Lidar 或相机等用于扫描匹配的传感器频率较低，通常几十赫兹）之间架起了桥梁，既实现了对运动畸变的精确校正，又为关键的扫描匹配优化步骤提供了高质量的初始位姿估计，是 Cartographer实现高精度和高鲁棒性局部 SLAM 的关键组件之一。

PoseExtrapolator主要依赖于imu数据或odometry数据进行位姿的初始估计，采用预计分的方式（可以通过循环调用imu_tracker中的advance实现单步积分的累积计算）
姿态外推器作为Local SLAM的主要类成员，为Local SLAM提供位姿的初始估计值。

PoseExtrapolator 提供的是一个高频的、短期的预测。
SLAM系统的局部优化（如Scan Matching）提供了中频的、基于当前环境的修正。
SLAM系统的全局优化（如回环检测和姿态图优化）提供了低频的、全局一致性的修正。

PoseExtrapolator 不断地从IMU/Odometry积分，但它会定期被来自SLAM后端（局部或全局优化结果）的更准确的位姿“拉回”到正确的轨道上。它不是一个独立的、盲目积分的模块，而是与SLAM优化过程紧密耦合的。如果没有这些来自优化的修订，PoseExtrapolator的输出会迅速漂移到不可用的程度。

```
void LocalTrajectoryBuilder2D::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(options_.use_imu_data()) << "An unexpected IMU packet was added.";
  InitializeExtrapolator(imu_data.time);
  extrapolator_->AddImuData(imu_data);
}
```

```
void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}
```
- TrimImuData函数用于从双端队列中修剪掉过时的imu数据。
```
//imu_data_.size() > 1 保证IMU队列中至少保留一个数据，避免队列完全清空
//!timed_pose_queue_.empty()确保存在有效的位姿时间作为修剪依据。
//imu_data_[1].time <= timed_pose_queue_.back().time 判断第二个IMU数据是否早于最新位姿时间。
//如果是，说明第一个IMU数据已过期，可删除。
//std::deque（双端队列）是C++标准模板库（STL）中提供的一种序列容器，
//支持在头部和尾部高效插入和删除元素，同时允许随机访问。
//std::deque<sensor::ImuData> imu_data_;
//动态维护最小必要的IMU数据队列，适应长时间运行场景。(过时的已经参与积分计算的imu数据将被删除)
void PoseExtrapolator::TrimImuData() {
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time) {
    imu_data_.pop_front();
  }
}
```
- `ExtrapolatePose`函数会根据时间估计出一个初始位姿姿态，位移估计给予最近的位移和线速度以及时间差进行，角度估计给予imu_tracker的角度增量估计和最近的角度的累积；
```
  struct TimedPose {
    common::Time time;
    transform::Rigid3d pose;
  };

//std::deque<TimedPose> timed_pose_queue_; timed_pose_queue_记录了通过局部位姿优化后的较为精确的位姿序列。
//cached_extrapolated_pose_保留time时刻姿势外推器预测的位姿信息
//ExtrapolatePose函数基于最近一些优化的位姿信息和当前时间和优化位姿的时间差及最近预估的线速度得出预估的位移(主要通过函数ExtrapolateTranslation计算估计)
//利用ExtrapolateRotation给定的参数的时间估计时间差你的旋转量，由于参数传递的是extrapolation_imu_tracker_，其在AddPose函数时会进行重新初始化
transform::Rigid3d PoseExtrapolator::ExtrapolatePose(const common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  if (cached_extrapolated_pose_.time != time) {
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

//参数imu_tracker在addpose时进行了初始化，
//odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
//extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
//因此对imu_tracker进行了预积分，但是函数体内的imu_tracker_还是原始的imu_tracker_，
//因此通过两个tracker之间的角度差异得出上次的位姿优化时间到当前时间time段内的积分角度
Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
    const common::Time time, ImuTracker* const imu_tracker) const {
  CHECK_GE(time, imu_tracker->time());
  AdvanceImuTracker(time, imu_tracker);
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  return last_orientation.inverse() * imu_tracker->orientation();
}
```

`AddPose`函数维护了最近一段时间内的优化的位姿，早期的会从队列中移出。

```
void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  if (imu_tracker_ == nullptr) {
    common::Time tracker_start = time;
    if (!imu_data_.empty()) {
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }
  timed_pose_queue_.push_back(TimedPose{time, pose});
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
  }
  UpdateVelocitiesFromPoses();
  AdvanceImuTracker(time, imu_tracker_.get());
  TrimImuData();
  TrimOdometryData();
  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
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
### 姿态外推器的pose只是局部优化的位姿和imu的预积分的结合吗，全局优化的位姿是否也会对其产生影响？
