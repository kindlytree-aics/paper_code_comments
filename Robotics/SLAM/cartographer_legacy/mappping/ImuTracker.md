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

```
//gravity_vector_(Eigen::Vector3d::UnitZ()),其物理含义是“重力方向沿 Z 轴”，而非“重力加速度值为 1 m/s²”。
//姿态估计的核心是 重力方向，而非重力大小。例如，在计算旋转四元数时，只需方向对齐，无需关心具体数值。
ImuTracker::ImuTracker(const double imu_gravity_time_constant,
                       const common::Time time)
    : imu_gravity_time_constant_(imu_gravity_time_constant),
      time_(time),
      last_linear_acceleration_time_(common::Time::min()),
      orientation_(Eigen::Quaterniond::Identity()),
      gravity_vector_(Eigen::Vector3d::UnitZ()),//其物理含义是“重力方向沿 Z 轴”，而非“重力加速度值为 1 m/s²”。
      imu_angular_velocity_(Eigen::Vector3d::Zero()) {}
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
//此函数通过EMA滤波从加速度计数据中提取重力分量，并动态调整姿态四元数，确保设备姿态与重力方向一致。
//代码逻辑紧密结合传感器特性，是惯性导航系统中姿态估计的核心模块。
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
  //越大的 delta_t 意味着当前值比旧值更可靠 → α 趋近于 1；，越大的delta说明当前时间比上次的时间差的较长
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

from deepseek r1：

当车体在二维平面（如x-y平面）加速时，IMU的线性加速度（`imu_linear_acceleration`）的z轴值**通常不会为0**，而是会包含重力分量（约9.8 m/s²）。此时，若车体保持水平姿态，加速度计的测量值由两部分组成：

1. **重力分量**：始终作用于加速度计的z轴（假设设备坐标系与车体坐标系对齐）。
2. **运动加速度**：由车体的运动产生，作用于x或y轴。

---

### **具体分析**

#### **1. 加速度计的测量原理**
加速度计测量的是**比力（specific force）**，即：
\[
\text{imu\_linear\_acceleration} = \text{运动加速度} - \text{重力加速度}
\]
- 当车体静止时，加速度计的z轴值为重力加速度（约9.8 m/s²），x/y轴接近0。
- 当车体在x-y平面加速时：
  - **x或y轴**：反映车体的运动加速度。
  - **z轴**：仍包含重力分量（除非车体处于自由落体或失重状态）。

#### **2. EMA对重力向量的影响**
假设车体在x轴加速，加速度计读数为：
\[
\text{imu\_linear\_acceleration} = [a_x, 0, 9.8]^\top \quad (\text{单位：m/s²})
\]
通过指数移动平均（EMA）更新重力向量：
\[
\text{gravity\_vector\_} = (1-\alpha) \cdot \text{gravity\_vector\_} + \alpha \cdot [a_x, 0, 9.8]^\top
\]
- **若车体短暂加速**（如瞬时运动）：
  - `a_x` 是短暂的高频信号，EMA会平滑掉大部分运动加速度，保留z轴的重力分量。
  - 最终 `gravity_vector_` 仍接近 `[0, 0, 9.8]`。
  
- **若车体持续加速**（如长时间匀加速）：
  - `a_x` 成为低频信号，EMA会将其部分吸收到 `gravity_vector_` 中。
  - 导致 `gravity_vector_` 的x轴出现非零值，**错误地将运动加速度当作重力分量**。

---

### **关键结论**
1. **z轴值通常不为0**：  
   除非车体处于自由落体，否则z轴始终包含重力分量（9.8 m/s²）。  
   **用户假设的“z轴为0”不符合实际物理场景**。

2. **EMA的滤波效果**：  
   - 若运动加速度是短时高频信号，EMA能有效滤除，`gravity_vector_` 仍以z轴为主。  
   - 若运动加速度是长时低频信号，EMA会将其误认为重力分量，导致 `gravity_vector_` 在x/y轴出现偏差。

### **示例场景**
#### **场景1：短暂加速**
- 加速度计测量值：`[2.0, 0, 9.8]`（车体在x轴加速2.0 m/s²）。
- 时间常数 `τ` 较大（如5秒），EMA平滑因子 `α` 较小。  
- **结果**：  
  `gravity_vector_` 的x轴略微增加（如从0变为0.1），但z轴仍接近9.8。

#### **场景2：持续加速**
- 加速度计长期测量值：`[2.0, 0, 9.8]`（车体持续匀加速）。
- 时间常数 `τ` 较小（如0.1秒），EMA平滑因子 `α` 较大。  
- **结果**：  
  `gravity_vector_` 的x轴逐渐收敛到2.0，z轴仍为9.8。  
  **重力方向被错误估计为倾斜**（x轴分量被误认为重力）。

---

### **解决方案**
1. **合理选择时间常数 `τ`**：  
   - 若车体运动频繁，需增大 `τ` 以抑制运动加速度对重力估计的影响。
   - 若环境稳定，可减小 `τ` 以快速响应真实重力变化。

2. **融合其他传感器**：  
   - 使用陀螺仪积分估计姿态，分离重力与运动加速度。
   - 结合GPS或轮速计数据，消除长期加速度偏差。

3. **运动状态检测**：  
   - 检测车体是否处于加速状态，动态调整EMA参数或暂停重力更新。


### **代码验证**
若强制将z轴设为0（模拟自由落体），EMA结果将完全错误：
```cpp
// 假设车体在x轴加速，且z轴人为设为0（错误场景）
imu_linear_acceleration = Eigen::Vector3d(2.0, 0, 0); 
// EMA更新后，gravity_vector_的x轴会逐渐积累，z轴衰减到0
```
此时 `gravity_vector_` 将偏离真实重力方向，导致姿态估计失效。

在斜坡且坡度不平的场景下，该代码的逻辑能够更好地体现其作用，同时通过 **重力向量估计** 和 **姿态旋转调整** 的耦合，实现动态校正。以下是详细分析：

### **1. 斜坡场景下的重力分量分解**
假设车体在斜坡上静止或运动，重力加速度在车体坐标系中的分量会发生变化：
- **水平路面**：重力完全作用于车体坐标系的Z轴（`gravity_vector_ ≈ [0, 0, 9.8]`）。
- **斜坡路面**：重力分解为垂直于斜坡的分量（Z轴）和平行于斜坡的分量（X或Y轴）。  
  **示例**：车体沿X轴方向上坡，倾斜角为θ，则重力分量为：
  \[
  \text{gravity\_vector\_} = [g \sin\theta, 0, g \cos\theta]^\top \quad (g=9.8 \, \text{m/s}^2)
  \]


### **2. EMA在斜坡场景中的作用**
#### **2.1 静止斜坡**
- **加速度计测量值**：仅包含重力分量，无运动加速度（假设静止）。  
  \[
  \text{imu\_linear\_acceleration} = [g \sin\theta, 0, g \cos\theta]^\top
  \]
- **EMA更新**：  
  通过指数移动平均平滑后，`gravity_vector_` 将收敛到真实的重力分量分解值。  
  **意义**：  
  - 重力向量的X轴分量反映斜坡的倾斜角度（\(\sin\theta\)）。  
  - 重力向量的Z轴分量反映垂直方向的重力分量（\(\cos\theta\)）。

#### **2.2 运动斜坡**
- **加速度计测量值**：包含重力分量和运动加速度。  
  \[
  \text{imu\_linear\_acceleration} = [g \sin\theta + a_x, 0, g \cos\theta]^\top
  \]
  其中 \(a_x\) 是车体的运动加速度。
- **EMA更新**：  
  - 若 \(a_x\) 是高频信号（如短暂加速），EMA会将其滤除，保留重力分量。  
  - 若 \(a_x\) 是低频信号（如持续匀加速），EMA会将其误认为重力分量，导致 `gravity_vector_` 的X轴出现偏差。  
  **风险**：斜坡倾角估计错误。

### **3. 姿态旋转的耦合校正**
代码中的以下部分通过旋转四元数，将估计的 `gravity_vector_` 对齐到车体坐标系的Z轴：
```cpp
const Eigen::Quaterniond rotation = FromTwoVectors(
    gravity_vector_, orientation_.conjugate() * Eigen::Vector3d::UnitZ());
orientation_ = (orientation_ * rotation).normalized();
```
#### **3.1 校正逻辑**
1. **目标对齐**：  
   将当前估计的 `gravity_vector_`（可能包含斜坡分量）旋转到与车体坐标系的Z轴对齐。  
   - 若车体在斜坡上静止，旋转后姿态四元数 `orientation_` 会反映斜坡的倾斜角度。  
   - 若车体在斜坡上运动，姿态会根据滤波后的重力方向动态调整。

2. **耦合关系**：  
   - **正向依赖**：`gravity_vector_` 的估计影响姿态调整。  
   - **反向依赖**：姿态调整后，新的 `orientation_` 会影响后续加速度计数据的坐标系转换（如将全局重力方向投影到车体坐标系）。

#### **3.2 斜坡不平的动态场景**
- **坡度变化**：  
  若车体在行驶中遇到坡度变化（如从平路进入斜坡），EMA会逐步更新 `gravity_vector_`，姿态旋转随之调整，最终反映新的斜坡倾角。
- **动态耦合示例**：  
  1. 车体开始上坡，加速度计检测到X轴分量增加（\(g \sin\theta\)）。  
  2. EMA逐步更新 `gravity_vector_`，X轴分量增大。  
  3. 旋转四元数调整姿态，使 `gravity_vector_` 对齐Z轴，此时 `orientation_` 反映上坡姿态。  
  4. 新的姿态下，加速度计的未来测量值将在更新后的坐标系中处理。


### **4. 代码在斜坡场景中的优势**
1. **重力方向动态跟踪**：  
   EMA能够适应斜坡倾角的缓慢变化（如逐渐上坡），而突变运动加速度会被滤除。  
   **示例**：  
   - 时间常数 \( \tau = 1 \, \text{s} \) 时，算法可在几秒内跟踪倾角变化。  
   - 若坡度突然变化（如驶过坑洼），EMA可避免过度响应。

2. **姿态与重力的自洽**：  
   通过强制 `gravity_vector_` 与姿态Z轴对齐，确保以下一致性：  
   - 姿态四元数 `orientation_` 始终反映当前重力方向。  
   - 加速度计测量值中的重力分量可被正确分离，用于运动估计。

3. **抗运动干扰能力**：  
   若车体在斜坡上短暂加速，EMA的高通特性可抑制运动加速度对重力估计的影响。


### **5. 潜在问题与改进**
#### **5.1 问题**
- **持续运动干扰**：  
  若车体在斜坡上长时间加速，EMA可能将运动加速度误认为重力分量，导致姿态倾斜角估计错误。
- **快速坡度变化**：  
  若坡度突变（如瞬间冲上陡坡），EMA的响应速度可能不足，导致姿态校正滞后。

#### **5.2 改进方案**
1. **多传感器融合**：  
   - 使用陀螺仪积分短期姿态变化，弥补加速度计的低频误差。  
   - 结合轮速计或视觉里程计，区分运动加速度与重力分量。  

2. **自适应时间常数**：  
   - 根据车体运动状态（静止/运动）动态调整 \( \tau \)：  
     - 静止时减小 \( \tau \)，快速收敛重力估计。  
     - 运动时增大 \( \tau \)，抑制加速度干扰。  

3. **运动加速度检测**：  
   - 检测加速度计数据的方差，若超过阈值则暂停重力更新，避免污染估计值。


### **6. 总结**
在斜坡且坡度不平的场景下，该代码通过以下机制发挥作用：
1. **重力向量估计（EMA）**：  
   - 分离静态重力分量与动态运动加速度。  
   - 自适应跟踪斜坡倾角变化。  
2. **姿态旋转耦合校正**：  
   - 确保姿态四元数与当前重力方向一致。  
   - 动态调整车体坐标系，抑制运动干扰。  

**最终效果**：  
- 在静态或缓变斜坡中，算法能准确估计倾角并校正姿态。  
- 在动态运动或坡度突变时，需结合其他传感器或策略提升鲁棒性。


### **总结**
- **z轴值通常包含重力**，EMA主要提取低频重力分量。  
- **持续运动加速度可能导致重力估计偏差**，需通过传感器融合或运动补偿解决。  
- 实际应用中需避免直接假设z轴为0，需结合物理约束和传感器特性设计算法。

### 问题： imu的线性加速度的读数中包含了重力的因素，但是实现在地面上时，车身的重力会在地面上给与一个反作用力，因此重力的因素对于运动没有影响，所以要减去这个重力的影响，才是实际影响运动的线性加速度？
answer from gemini
您理解得非常正确，这正是姿态估计和运动分析中的核心环节！让我们更清晰地梳理一下：

1.  **IMU 加速度计的读数**：
    *   加速度计测量的是**比力 (Specific Force)**。通俗地讲，它测量的是设备感受到的所有力（除了引力本身）除以质量所产生的加速度。
    *   更精确的公式是：`a_IMU_reading = a_motion - g_vector_in_sensor_frame`
        *   `a_IMU_reading`：加速度计的直接读数。
        *   `a_motion`：设备由于所有**非引力**（如引擎推力、制动力、摩擦力、空气阻力、地面支撑力等合力）产生的**真实线性加速度**。这才是我们通常关心的“运动加速度”。
        *   `g_vector_in_sensor_frame`：真实的**重力加速度矢量**在**当前传感器坐标系**中的表示。例如，如果世界坐标系的Z轴向下为正，重力为 `[0, 0, +9.8]` m/s²，而传感器此时是水平的且其Z轴向上，那么经过姿态旋转后，`g_vector_in_sensor_frame` 可能近似为 `[0, 0, -9.8]` m/s²。

2.  **地面上的情况与反作用力**：
    *   当车身静止或在水平地面上匀速行驶时，`a_motion` 在垂直方向上为零。
    *   此时，车受到向下的重力 `mg` 和地面向上的支撑力 `N`（反作用力）。因为垂直方向没有加速度，所以 `N = mg`。
    *   加速度计“感受”到的是这个支撑力 `N` 作用在质量 `m` 上产生的效果。所以，如果传感器的Z轴垂直向上，它会读出一个向上的加速度约 `+9.8 m/s²`。
    *   代入公式：
        *   `a_motion` (垂直分量) = 0
        *   `g_vector_in_sensor_frame` (Z分量) = -9.8 m/s² (假设传感器Z轴向上，重力向下为正)
        *   `a_IMU_reading` (Z分量) = 0 - (-9.8 m/s²) = +9.8 m/s²。这与我们的直觉相符。

3.  **“减去重力的影响”的含义**：
    *   您说的“重力的因素对于运动没有影响”，在“垂直方向净运动”这个层面上是对的，因为地面支撑力抵消了它。
    *   但是，重力**始终存在**，并且加速度计**始终能间接测量到它**（通过测量抵抗重力所需的力，或者自由落体时测量不到力）。
    *   为了得到**实际影响运动的线性加速度 `a_motion`**，我们需要从加速度计的读数中分离出由重力引起的部分。
    *   从公式 `a_IMU_reading = a_motion - g_vector_in_sensor_frame` 变换得到：
        `a_motion = a_IMU_reading + g_vector_in_sensor_frame`

4.  **如何实现“减去”**：
    *   关键在于准确知道 `g_vector_in_sensor_frame`。
    *   我们知道在**世界坐标系（或地理坐标系）**中，重力加速度是一个已知的常量矢量（例如，`[0, 0, -9.8]` m/s²，如果Z轴向上）。
    *   ImuTracker（姿态估计算法）的核心任务之一就是估算出传感器当前相对于世界坐标系的**姿态**（通常用旋转矩阵 `R_world_to_sensor` 或四元数表示）。
    *   一旦知道了姿态，就可以将世界坐标系下的重力矢量转换到传感器坐标系下：
        `g_vector_in_sensor_frame = R_world_to_sensor * g_vector_in_world_frame`
    *   然后，就可以计算出真正的运动加速度：
        `a_motion = a_IMU_reading + (R_world_to_sensor * g_vector_in_world_frame)`

**总结您的理解：**

*   **正确**：IMU的线性加速度读数中确实“包含”了重力的因素（或者说，反映了与重力相关的力）。
*   **正确**：在地面上，重力被地面的反作用力（支撑力）平衡，所以物体在垂直方向上没有因重力而产生的净加速度。
*   **核心正确**：为了得到真正描述物体运动状态的加速度（由引擎、刹车、转向等产生的加速度），我们必须从IMU的读数中“减去”重力加速度在当前传感器姿态下的投影。

这个过程通常被称为**重力补偿 (Gravity Compensation)**。它是IMU数据处理中非常关键的一步，无论是在机器人导航、人体运动分析还是无人机飞行控制中都至关重要。


关于IMU预积分的相关计算，相关的实现也可以参考[FAST-LIVO2](https://gitee.com/kindlytree/fast-livo2-comments/blob/main/src/IMU_Processing.cpp)中的实现。

