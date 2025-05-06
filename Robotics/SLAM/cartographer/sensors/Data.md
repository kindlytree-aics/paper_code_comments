# sensor/Data

## 传感器数据抽象基类（核心设计）
- 作用：
定义所有传感器数据的通用接口，强制子类实现关键方法（GetTime()、AddToTrajectoryBuilder()），统一多传感器数据的访问方式。

- 例如在`cartographer\mapping\internal\2d\local_slam_result_data.h`中

```c++
 common::Time GetTime() const override { return time_; }
  virtual void AddToPoseGraph(int trajectory_id,
                              PoseGraph* pose_graph) const = 0;
```
就会根据当前所用到的data重写这两个函数


- 系统支持：
为激光雷达、IMU、里程计等不同传感器提供统一的数据抽象层，简化后续处理逻辑（如轨迹构建、数据融合），增强系统扩展性。

## 传感器标识管理（sensor_id_）
- 作用：
通过GetSensorId()返回传感器唯一标识符，标识数据来源。

- 系统支持：
支持多传感器配置（如多激光雷达或混合传感器系统），确保数据来源可追溯，便于传感器标定、数据关联和故障排查。

## 数据时间戳获取（GetTime()）
- 作用：
提供数据采集的精确时间（common::Time），用于时间同步。

- 系统支持：
实现多传感器数据的时间对齐（如激光雷达与IMU同步），为运动畸变校正、传感器融合（如卡尔曼滤波）提供时间基准，提升定位精度。

## 数据注入接口（AddToTrajectoryBuilder()）
- 作用：
将传感器数据传递给轨迹构建器（TrajectoryBuilderInterface），触发数据处理流程。

- 系统支持：
作为数据流驱动的核心机制，实现传感器数据到SLAM算法的主动推送。不同传感器数据（如点云、惯性数据）通过重写此方法调用轨迹构建器的对应处理接口（如,`local_slam_result_2d.h的AddToTrajectoryBuilder`），解耦数据采集与算法逻辑。