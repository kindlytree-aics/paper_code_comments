## point_cloud源码分析

- 文件路径 `/cartographer/sensor/point_cloud.cc point_cloud.h`

 包含了
 `
 "Eigen/Core"
 "glog/logging.h"
`的轻量级第三方库,Eigen用于进行相关数学代数计算,glog用于进行记录日志


功能列表及分析

1. 点云数据结构定义

在`rangefinder_point.h`中,定义了许多基础rangefinder数据格式以及相关操作符重载和Proto序列化反序列化函数,如下

```c++
// Stores 3D position of a point observed by a rangefinder sensor.
struct RangefinderPoint {
  Eigen::Vector3f position;
};

// Stores 3D position of a point with its relative measurement time.
// See point_cloud.h for more details.
struct TimedRangefinderPoint {
  Eigen::Vector3f position;
  float time;
};

...
```

由于每一帧激光点云数据中的点是在不同时刻采集的（激光线束扫描时的发出的rays的时间不同），因此每一个点都带有测量时间信息。

每一帧的点云的点集合的时间不同，因此存在着畸变，需要根据信息将点的空间信息按时间进行对齐（运动补偿，去畸变， FAST-LIVO2中基于imu的去畸变已经介绍过）。

TimedRangefinderPoint 的集合（例如 std::vector<TimedRangefinderPoint> 或类似的结构）作为其内部处理来自 LaserScan 和 PointCloud2 数据的标准化格式，从而能够应用其核心 SLAM 算法，有效处理运动失真等问题。


在`sensor/point_cloud.h`中定义了基础的PointCloud类型`std::vector<PointType> points`
,其中PointType为
```c++
 using PointType = RangefinderPoint;
```
和扩展结构PointCloudWithIntensities，用于存储三维点云及其关联的强度信息。

系统支持：
为传感器数据提供统一的内存表示，是后续处理（如坐标变换、滤波、匹配）的基础数据容器。强度信息可用于增强特征提取或环境建模。

1. 点云坐标变换 (TransformPointCloud)
作用：
对点云应用刚体变换（旋转+平移），将点云从传感器坐标系转换到其他坐标系（如全局坐标系）。

系统支持：
支撑传感器数据与地图的坐标对齐，是SLAM中多帧数据融合、运动补偿和地图拼接的核心操作。例如，将激光雷达数据根据机器人的实时位姿转换到全局坐标系中。

3. 点云裁剪 (Crop)
作用：
根据Z轴范围（min_z和max_z）过滤点云，移除超出范围的点。

系统支持：
预处理阶段去除无效数据（如地面反射、天花板干扰），降低计算复杂度，提升后续算法（如点云匹配、特征提取）的效率和鲁棒性。

4. 序列化与反序列化 (FromProto,ToProto)
作用：
将点云与Protobuf格式互相转换，实现数据的持久化存储或网络传输。

系统支持：
支持离线地图保存、跨模块数据传输（如从传感器驱动到建图模块），便于调试和分布式系统部署。例如，保存点云用于回放测试或地图编辑。

5. 测试用例 (point_cloud_test.cc)
作用：
验证TransformPointCloud在绕Z轴旋转90度时的正确性，确保坐标变换逻辑符合几何预期。

系统支持：
保障核心功能的正确性，防止因变换错误导致地图错位或定位失效，提升系统整体可靠性。