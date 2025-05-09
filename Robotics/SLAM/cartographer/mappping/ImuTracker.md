# ImuTrack实现分析

IMU传感器提供两个关键信息
- 角速度（gyroscope）：估计姿态变化（积分求得 orientation）；
- 线性加速度（accelerometer）：估计线性运动，但也包含了重力加速度的分量。静止状态下，观测值等于重力加速度，加速度计就可以被用来估计当前重力方向（朝下），进一步用于姿态校正（主要是 pitch 和 roll）
- IMU读取的加速度（测量值）是线性加速度和重力加速度的合加速度

但做自由落体运动时，加速度计读数为0（内部的测试质量都在自由落体，没有非引力作用在测试质量上，所以加速度计的读数是 0。）
因此a_actual = a_measured + g_imu

IMU 的加速度计测量值实际上是：加速度计测量值 = 实际线性加速度 - 重力加速度矢量

$$
\mathbf{a}_{\text{measured}} = \mathbf{R}^\top \cdot (\mathbf{a}_{\text{true}} - \mathbf{g})
\quad \text{⇔} \quad 
\mathbf{a}_{\text{true}} = \mathbf{R} \cdot \mathbf{a}_{\text{measured}} + \mathbf{g}
$$

但注意！上面这行是指**加速度计的测量是世界系下加速度减去重力后，旋转到IMU系**。

```
  const double imu_gravity_time_constant_;
  common::Time time_; //imu_tracker 当前状态对应的时间戳。调用 Advance() 时更新。
  //最后一次调用 AddImuLinearAcceleration() 的时间。
  //用于计算两次线性加速度更新之间的时间间隔 dt，供滤波器使用（如计算 alpha）
  common::Time last_linear_acceleration_time_;
  //orientation_,当前估计的姿态（旋转，世界系到IMU系）。
  //通过 IMU 的角速度 AddImuAngularVelocity() 不断用积分方式更新：
  //保持一个实时更新的旋转状态供系统调用
  Eigen::Quaterniond orientation_;
  //通过线性加速度低通滤波估计出的重力方向
  Eigen::Vector3d gravity_vector_;
  //姿态的积分在 Advance(time) 中完成。
  Eigen::Vector3d imu_angular_velocity_;

```

## 状态积分逻辑实现

旋转的变化量为角速度乘以时间（较短的时间认为角速度匀速）。
如何更好的理解重力向量的作用？

```
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


```
void ImuTracker::AddImuLinearAccelerationObservation(
    const Eigen::Vector3d& imu_linear_acceleration) {
  // Update the 'gravity_vector_' with an exponential moving average using the
  // 'imu_gravity_time_constant'.
  //如果有前一次加速度观测，就用当前时间与上次时间差作为 delta_t；
  //否则认为这是第一次观测，delta_t 为无穷大（这样会完全采用当前加速度，下面会看到）。
  const double delta_t =
      last_linear_acceleration_time_ > common::Time::min()
          ? common::ToSeconds(time_ - last_linear_acceleration_time_)
          : std::numeric_limits<double>::infinity();
  last_linear_acceleration_time_ = time_;
  //α 是一个与 delta_t 和 imu_gravity_time_constant_ 有关的系数；
  //越大的 delta_t 意味着当前值比旧值更可靠 → α 趋近于 1；
  //越小的 delta_t → α 趋近于 0，表示保留更多旧的重力估计；
  //指数加权移动平均 (exponential moving average) 的标准写法
  const double alpha = 1. - std::exp(-delta_t / imu_gravity_time_constant_);
  gravity_vector_ =
      (1. - alpha) * gravity_vector_ + alpha * imu_linear_acceleration;
  // Change the 'orientation_' so that it agrees with the current
  // 'gravity_vector_'.
  //FromTwoVectors(a, b) → 计算从 a 旋转到 b 的旋转四元数；
  const Eigen::Quaterniond rotation = FromTwoVectors(
      gravity_vector_, orientation_.conjugate() * Eigen::Vector3d::UnitZ());
  //最终作用：通过 rotation 把当前姿态旋转到与重力方向一致（pitch 和 roll 对齐，yaw 不变）；
  orientation_ = (orientation_ * rotation).normalized();
  CHECK_GT((orientation_ * gravity_vector_).z(), 0.);
  CHECK_GT((orientation_ * gravity_vector_).normalized().z(), 0.99);
}
```

关于IMU预积分的相关计算，相关的实现也可以参考[FAST-LIVO2](https://gitee.com/kindlytree/fast-livo2-comments/blob/main/src/IMU_Processing.cpp)中的实现。

