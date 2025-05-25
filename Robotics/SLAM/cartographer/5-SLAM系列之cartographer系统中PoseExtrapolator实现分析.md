

本篇文章将向大家介绍cartographer中的姿态外推器PoseExtrapolater的实现，姿态外推器主要基于ImuData和/或OdometryData以及局部SLAM优化后的位姿实现较近的未来时刻的车体(机器人本体)的姿态估计。
下面将和代码相结合进行较为详细具体的分析介绍。

姿态外推器的主要作用体现在：
- 1、在同步后的点云数据的每一个点云点上都会调用ExtrapolatePose函数预估出即时姿态，
以便基于点云坐标的去畸变操作(将点云点相对于车体坐标系的坐标和车体姿态叠加获取点云点在统一坐标系下的坐标从而实现了去畸变的操作，和FAST-LIVO2中的运动补偿实现去畸变的思路和实现方法不同)；
- 2、当点云帧的最后一个点的坐标校正完成后调用ScanMatch函数对基于外推器预测的姿态和点云数据基础上进行位姿优化获得更加精确的姿态，同时调用extrapolator_->AddPose函数将优化后的姿态更新到姿态外推器，姿态外推器基于更加准确的姿态的基础上进行后续时间点的位姿估计。如下图所示，在帧内和帧的边界分别调用了姿态外推器的上述两个函数。

图1：姿态外推器的位姿估计函数调用方法
下面就具体的实现代码细节做一些说明。

PoseExtrapolater的成员变量及其含义和作用见下面的代码和对应的注释说明。

```
//支持一定时间段的位姿队列长度，时间较为久远一点的数据将被释放
const common::Duration pose_queue_duration_;
//用于记录时间序列对应的优化后位姿，采用双端队列(支持push_back和pop_front)，
//从队尾插入排队,从队列前pop释放早时刻的位姿数据；方便保持合适的存储空间长度
//通过当前最近的位姿和此后imu的预积分得出的旋转角度增量估计旋转角度
//通过当前最近的位姿和估计的线速度以及时间跨度估计位移
std::deque<TimedPose> timed_pose_queue_;
//从相邻位姿的变化估计线速度和角速度
Eigen::Vector3d linear_velocity_from_poses_ = Eigen::Vector3d::Zero();
Eigen::Vector3d angular_velocity_from_poses_ = Eigen::Vector3d::Zero();

const double gravity_time_constant_;//用于imu中的重力向量滤波计算
std::deque<sensor::ImuData> imu_data_;
//用于整体姿态的预积分计算
std::unique_ptr<ImuTracker> imu_tracker_;
std::unique_ptr<ImuTracker> odometry_imu_tracker_;
//用于点云点的旋转姿态估计
std::unique_ptr<ImuTracker> extrapolation_imu_tracker_;
TimedPose cached_extrapolated_pose_;

std::deque<sensor::OdometryData> odometry_data_;
Eigen::Vector3d linear_velocity_from_odometry_ = Eigen::Vector3d::Zero();
Eigen::Vector3d angular_velocity_from_odometry_ = Eigen::Vector3d::Zero();

```

AddPose函数在局部SLAM算法优化位姿后调用，通过ScanMatch算法优化后的位姿和对应的时间作为参数调用AddPose函数，具体关于AddPose函数的实现细节及分析请参考如下的代码和注释。

```
void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
if (imu_tracker_ == nullptr) {//如果没有初始化imu_trakcer_则进行初始化
  common::Time tracker_start = time;
  if (!imu_data_.empty()) {
    tracker_start = std::min(tracker_start, imu_data_.front().time);
  }
  imu_tracker_ =  absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
}
//将time时刻的位姿加入到时间位姿元组队列中
timed_pose_queue_.push_back(TimedPose{time, pose});
//判断如果队列的前端的位姿时刻比设定的时间跨度阈值还要晚，则释放掉一些"过时"的时间位姿元组
while (timed_pose_queue_.size() > 2 &&
    timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
}
//UpdateVelocitiesFromPoses函数根据pose_queue_duration_时间跨度和首 
//尾的位姿估计平局线速度linear_velocity_from_poses_和平均角速度angular_velocity_from_poses_
UpdateVelocitiesFromPoses();
  
//AdvanceImuTracker函数将imu_tracker_根据time时间参数向前积分
//根据imu_data_队列中小于time时刻的imu数据以及调用imu_trakcer_的相关方 
//法进行向前逐项单步预积分累积计算实现time时刻旋转姿势的估计，
//在这里主要是进行预积分的计算，将位姿中的旋转部分分量随着时间的积分结果存放到  
//imu_tracker的orientation相关变量，前篇文章已经介绍过，
//imu_trakcer主要对位姿中的旋转部分进行估计
//位移通过姿态外推器中估计的线速度(假设以匀速运动)和时间跨度的乘积进行估计
//imu_tracker的旋转状态积分到time时间，为下一个的time时间提供起初值，提供给后面初始化的两个tracker
//用于估计下一个点云帧的每一个点云点点的位姿估计，以便进行去畸变的运算
AdvanceImuTracker(time, imu_tracker_.get());
//对于已经使了的imu_data和odometry_data的历史数据释放其存储空间
TrimImuData();
TrimOdometryData();
//将此时刻的imu_tracker对象复制到到成员变量odometry_imu_tracker_和extrapolation_imu_tracker_
//中便于后面的旋转增量的计算。这两个变量在每次AddPose函数被调用时都会基于imu_tracker_进行重新初始化
//extrapolation_imu_tracker_用于估计旋转角度变化增量
//度变化增量代码表现为：last_orientation.inverse() * imu_tracker->orientation();
odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}
```

关于姿态外推器中的旋转和位移估计的代码实现，下面的段落文字给出具体的代码和分析注释。

```
//调用AdvanceImuTracker函数根据时间对imu数据求预积分获取位姿中的旋转部分
//需要注意的是imu_tracker是作为参数传入的，根据实际情况
//其有可能是extrapolation_imu_tracker_和odometry_imu_tracker_
//extrapolation_imu_tracker_主要用于旋转增量估计
//旋转估计为基于局部优化的位姿和旋转增量估计
Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
 const common::Time time, ImuTracker* const imu_tracker) const {
 CHECK_GE(time, imu_tracker->time());
 AdvanceImuTracker(time, imu_tracker);
 const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  return last_orientation.inverse() * imu_tracker->orientation();
}
//位移的估计基于时间跨度和估计出的线性速度(在该时间段内认为是匀速的)
//位移估计没有基于imu的线性加速度等估计，imu的线性加速度的累积漂移现象更加明显
Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  if (odometry_data_.size() < 2) {
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  return extrapolation_delta * linear_velocity_from_odometry_;
}
```

References

[1]、 姿态外推器的实现：./cartographer/mapping/pose_extrapolator.cc