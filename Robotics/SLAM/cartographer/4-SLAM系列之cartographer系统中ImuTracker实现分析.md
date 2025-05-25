# 4-SLAM系列之cartographer系统中ImuTracker实现分析

这篇文章将向大家介绍cartographer系统中姿态外推器PoseExtrapolator实现分析。姿态外推器可以看成是一种软件逻辑实现的里程计，但是其利用了局部位姿优化的结果及时更新更高精度的位姿，从而有效防止出现位姿的严重累积漂移现象。本篇文章将从两个方面进行介绍，第一部分介绍IMU传感器的基本作用及采用高频ImuData进行预积分估计姿态的方法(主要体现在ImuTracker类里)；第二部分将介绍姿态外推器和局部位姿优化的结果如何结合实现更精确的位姿估计(主要体现在PoseExtrapolator类里)。

惯性测量单元（IMU, Inertial Measurement Unit）传感器包含陀螺仪(主要测量角速度angular_velocity，频率高)和加速度计(主要测量linear_acceleration，频率低)。但在IMU的硬件驱动层一般会对低频率的线性加速度值进行插值或直接填充，以保持和角速度的测量同样的频率输出，因此输出的消息同时有这两个测量值。IMU的消息标准中msg->header.frame_id 是 sensor_msgs::Imu 消息中声明的IMU数据所在的坐标系名称（例如 “imu_link”）,IMU和 tracking_frame 必须共址（The IMU frame must be colocated with the tracking frame，否则由于imu和tracking_frame的位置上平移关系的存在导致车体的线性加速度的估计变得困难），以便IMU相关算法简化计算。

IMU的线性加速度的读数值为一个三维的向量，测量的是比力（specific force），加速度计的测量值里天然包含重力方向的加速度分量(比如车体在平面上静止时IMU的加速度计读数为[0,0,9.8]，当车体做自由落体运动时，加速度计的读数为0。

```
//Imutracker的构造函数初始化
//重力向量初始化为标准Z轴，这里主要强调的是方向，没有显示定义具体的大小，如9.8m/s²。
//imu_gravity_time_constant为重力时间常数，可以在配置文件如cartographer\configuration_files\trajectory_builder_2d.lua中进行定义。
ImuTracker::ImuTracker(const double imu_gravity_time_constant,
                       const common::Time time)
    : imu_gravity_time_constant_(imu_gravity_time_constant),
      time_(time),
      last_linear_acceleration_time_(common::Time::min()),
      orientation_(Eigen::Quaterniond::Identity()),
      gravity_vector_(Eigen::Vector3d::UnitZ()),//其物理含义是“重力方向沿 Z 轴”，而非“重力加速度值为 1 m/s²”。
      imu_angular_velocity_(Eigen::Vector3d::Zero()) {}
```

Imu在做time时刻的位姿估计的时候，其主要估计的是旋转的部分(位姿估计的平移部分主要在姿态外推器中利用预估的线速度和时间差进行估计)。

```
//在现有的位姿orientation_上叠加位姿rotation
//四元数只处理表示旋转(orientation)，不表示平移(translation);
//四元数相乘等于旋转的叠加；
//rotation.conjugate()四元数的共轭，代表rotation的逆旋转
//rotation.conjugate() * gravity_vector_表示在delta_t实践内车身的旋转导致的重力向量的变化；
//随着车身的旋转（比如在一个 delta_t 时间段内），重力在车身坐标系下的“样子”确实会改变。例如，如果车头从水平抬起，那么原本在车身看来是“向下”的重力，会逐渐有一个“向后”的分量。
void ImuTracker::Advance(const common::Time time) {
  CHECK_LE(time_, time);
  const double delta_t = common::ToSeconds(time - time_);
  const Eigen::Quaterniond rotation =
      transform::AngleAxisVectorToRotationQuaternion(
          Eigen::Vector3d(imu_angular_velocity_ * delta_t));
  orientation_ = (orientation_ * rotation).normalized();
  gravity_vector_ = rotation.conjugate() * gravity_vector_;
  time_ = time;
}
```

如上ImuData所述，ImuTracker一般会同时接收到角速度和线性加速度的消息，接收角速度消息时，直接更新角速度变量的值即可(用于后续基于此进行角度增量预估)，在接收线性加速度时的处理稍微复杂一些，下面是具体的分析。

```
//采用指数滑动平均，delta越大，alpha就越大，重力向量请更偏重于当前测量的线性加速度的值，否则和之前的重力向量保持更多的相似性
//指数滑动平均是一种低通滤波，主要是较为稳定计算gravity_vector_随着时间的变化
//下面的代码主要的任务逻辑主要有：1、估计出新时刻车体的重力向量方向；2、根据新的重力方向估计出角度变化增量rotation,并基于此
//增量预估出新得旋转角度，并将结果和重力向量得方向一致性再次进行验证；
void ImuTracker::AddImuLinearAccelerationObservation(
  const Eigen::Vector3d& imu_linear_acceleration) {
// Update the 'gravity_vector_' with an exponential moving average using the 'imu_gravity_time_constant'.
  const double delta_t =
      last_linear_acceleration_time_ > common::Time::min()
          ? common::ToSeconds(time_ - last_linear_acceleration_time_)
          : std::numeric_limits<double>::infinity();
last_linear_acceleration_time_ = time_;
const double alpha = 1. - std::exp(-delta_t/imu_gravity_time_constant_);
gravity_vector_ = (1. - alpha) * gravity_vector_ + alpha * imu_linear_acceleration;
  // Change the 'orientation_' so that it agrees with the current
  // 'gravity_vector_'.
const Eigen::Quaterniond rotation = FromTwoVectors(
      gravity_vector_, orientation_.conjugate() * Eigen::Vector3d::UnitZ());
  orientation_ = (orientation_ * rotation).normalized();
  CHECK_GT((orientation_ * gravity_vector_).z(), 0.);
  CHECK_GT((orientation_ * gravity_vector_).normalized().z(), 0.99);
}
```

一般的使用方式如下(在姿态外推器中调用），通过旋转姿态估计函数Advance和更新线性加速度观测值值函数AddImuLinearAccelerationObservation实现了旋转角度和重力向量的双重校正。实现了旋转姿态和重力向量的双重校正，使用IMU的陀螺仪测得的角速度和线性加速度计测得的线性加速度两个观测值相互校正和纠偏，实现更加稳定的姿态估计。

```
//时间向前预估新的旋转角度并更新校正重力向量方向
imu_tracker->Advance(it->time);
//更新线性加速的的观测值，更新重力向量，同时对旋转方向进行校正
imu_tracker->AddImuLinearAccelerationObservation(
    it- >linear_acceleration);
//更新角速度的值，为预估下一个时间点提供处事角速度值
imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
```

References

[1]、ImuTracker的实现: ./cartographer/mapping/imu_tracker.cc
[2]、PoseExtrapolater的实现：./cartographer/mapping/pose_extrapolator.cc