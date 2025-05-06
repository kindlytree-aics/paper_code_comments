## RangeData源码分析

统一表示传感器原始数据，为后续处理（如坐标变换、地图构建）提供结构化输入，是SLAM算法的基础数据单元。

原始带时间戳的测量点 (TimedRangefinderPoint) 已经被用于执行运动补偿（Motion Compensation）和去畸变（Deskewing）。  
RangeData 中存储的点云是补偿和去畸变后的结果，所有点都被转换到了与 RangeData 的 origin 相对应的统一参考系和时间点下（为不太时间信息的PointCloud）。


封装单次传感器扫描数据，包含：

```c++
struct RangeData {
  Eigen::Vector3f origin;
  PointCloud returns;
  PointCloud misses;
};
```

origin：传感器原点坐标（Eigen::Vector3f）。
returns：检测到的障碍物点云（PointCloud）,有返回
misses：未检测到障碍物的自由空间点云（PointCloud）。没有返回，可以理解为ray发射到了空旷空间

## 序列化与反序列化（ToProto/FromProto）
- 作用：
将RangeData与Protobuf格式互相转换，支持数据持久化存储和跨模块传输。

系统支持：
实现离线地图保存（如保存为文件）、多机器人协同（网络传输）、调试数据记录，增强系统模块化和跨平台兼容性。

## 坐标变换（TransformRangeData）

```c++
RangeData TransformRangeData(const RangeData& range_data,
                             const transform::Rigid3f& transform)
```

- 作用：
对RangeData中的origin、returns和misses应用刚体变换（Rigid3f），将其从传感器坐标系转换到目标坐标系（如机器人或全局坐标系）。

- 系统支持：
支撑多传感器数据融合（如IMU与激光雷达数据对齐）、运动畸变校正（补偿传感器运动），确保数据在统一坐标系下处理，提高地图一致性。


## 点云裁剪（CropRangeData）
- 作用：
根据Z轴范围（min_z和max_z）过滤returns和misses点云，移除无效点（如地面反射或噪声）。

- 系统支持：
提升数据质量，减少后续算法（如点云匹配、特征提取）的计算量，避免无效数据干扰定位和建图。


--

以下功能在最新代码中被删除,虽在 `range_data.h`引用了
```c++
include "cartographer/sensor/compressed_point_cloud.h"`
```
但未有相关实现

## 数据压缩与解压（CompressedRangeData）

- 作用：
将RangeData中的点云压缩为CompressedPointCloud，降低存储和传输开销；解压恢复为原始格式。

- 系统支持：

优化资源使用：
存储：长期地图存储节省磁盘空间（如大型环境建图）。
传输：减少网络带宽占用（分布式SLAM系统）。
内存：按需解压（迭代器访问），避免全量加载。

##   压缩数据转换（Compress/Decompress）

- 作用：
在RangeData与CompressedRangeData间转换，保持数据逻辑一致性。

- 系统支持：
无缝集成压缩与解压流程，允许系统在不同阶段灵活选择数据格式（如在线处理用原始数据，存储用压缩数据）。