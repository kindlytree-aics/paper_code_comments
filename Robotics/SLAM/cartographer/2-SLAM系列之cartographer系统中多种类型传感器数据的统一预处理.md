# SLAM系列之cartographer系统中多种类型传感器数据的统一预处理


在cartographer系统中，激光点云作为定位和建图的核心数据，在系统中不同的类对象中的流转和变换(包括坐标变换和数据类型的变换)是最频繁和复杂的，这篇文章将向大家梳理一下cartographer系统中多种类型传感器数据的统一预处理，包括传感器数据类型从标准ROS消息到cartographer内部处理所需数据类型的转换，消息多队列缓存和分发派遣机制等，如对本文理解和分析的过程有疑问或发现有问题的地方欢迎联系作者讨论并校正，谢谢。

1、首先在cartographer_ros的节点中，接收到的点云消息为标准通用的点云消息定义（根据物理传感器的不同，点云的标准消息有三类: sensor_msgs::PointCloud2，sensor_msgs::LaserScan，sensor_msgs::MultiEchoLaserScan，sensor_msgs 包用来专门存放消息的标准定义格式，其中msg.header.stamp定义了点云帧扫描的起始时间)，ROS节点收到点云消息后，在SensorBridge类对象中调用了三个函数名相同(函数名均为ToPointCloudWithIntensities，具体的实现参考http://msg_conversion.cc代码)但参数不同的重载函数将上述三种不同的点云数据转换成为一致的cartographer内部的PointCloudWithIntensities类型对象，该类型对象结构含有成员变量points和intensities，其中points的每一个点云点含有时间信息（TimedCloudPoint，时间为相对于点云扫描帧结束时间的偏移，为负值），intensities为点云点对应的强度信息，为浮点值数组。 以sensor_msgs::PointCloud2点云消息为例，SensorBridge类的函数HandlePoint2Message对接收到的消息进行处理，其中会通过函数调用ToPointCloudWithIntensities(*msg)进行的转换，在sensor_msgs::PointCloud2类型转换的工程中，用到了较为常用的pcl点云处理库，先将数据转换成pcl::PointCloud<PointXYZT>数据类型（在pcl::PointCloud数据类型汇总，点云点的时间戳为相对于点云帧扫描开始的时间偏移，为正值），然后基于该类型数据转换为PointCloudWithIntensities类型。

2、SensorBridge中通过函数HandleRangefinder将TimedPointCloud数据（PointCloudWithIntensities的points成员变量）转化为TimedPointCloudData类型的数据对象，该转换涉及到了点云数据从传感器坐标系到机器人本体坐标系（车身坐标系）的坐标变换，该变换定义为sensor_to_tracking->cast<float>()，而tracking_frame一般在lua配置文件中定义为“base_link”（一般为车体底盘后面中间位置）。同时TimedPointCloudData的time成员变量值为点云帧扫描结束时间，其origin成员变量定义了Lidar传感器相对于tracking_frame的位置偏移（ sensor_to_tracking->translation().cast<float>()）。将转换后的数据通过调用trajectory_builder_->AddSensorData转入到cartographer的核心库进行后续的算法处理。具体的变换代码为：

```
trajectory_builder_->AddSensorData(
        sensor_id, carto::sensor::TimedPointCloudData{
                 time, sensor_to_tracking->translation().cast<float>(),
                       carto::sensor::TransformTimedPointCloud(
                           ranges, sensor_to_tracking->cast<float>())});
```

3、上面的trajectory_builder_变量的类型为CollatedTrajectoryBuilder，在MapBuilderBridge::AddTrajectory函数里通过调用map_builder_->AddTrajectoryBuilder函数来生成，在http://map_builder.cc代码文件里的实现片段可供参考：

```
trajectory_builders_.push_back(absl::make_unique<CollatedTrajectoryBuilder>(
        trajectory_options, sensor_collator_.get(), trajectory_id,
        expected_sensor_ids,
        CreateGlobalTrajectoryBuilder2D(
            std::move(local_trajectory_builder), trajectory_id,
            static_cast<PoseGraph2D*>(pose_graph_.get()),
            local_slam_result_callback, pose_graph_odometry_motion_filter)));
```

其中map_builder对象在http://node_main.cc文件里通过调用cartographer::mapping::CreateMapBuilder(node_options.map_builder_options)进行的初始化），CollatedTrajectoryBuilder类为local_trajectory_builder（局部SLAM，前端）和PoseGraphND（全局SLAM，后端）的对象的集成封装，为SLAM前后端的集成入口。trajectory_builders_为vector数组容器，因此cartographer系统默认支持多个trajectory的构建。在cartographer核心库的实现中，不同的传感器数据将会在一起集中进行队列缓存管理和进一步分发。相关的数据队列缓存管理和分发的对象为sensor_collator_ ，其内部实现中使用了多个队列来分别缓存不同的传感器消息，并基于类似归并排序的思想从多个队列中取出最早的sensor数据进行分发(Dispatch)。

```
//map_builder.cc的构造函数中sensor_collator_成员变量的初始化
if (options.collate_by_trajectory()) {
  sensor_collator_ = absl::make_unique<sensor::TrajectoryCollator>();
}else {
  sensor_collator_ = absl::make_unique<sensor::Collator>();
}

//HandleCollatedSensorData为OrderedMultiQueue的callback函数
//当某一帧传感器数据被Dispatch的时候callback函数会被触发调用
sensor_collator_->AddTrajectory(
      trajectory_id, expected_sensor_id_strings,
      [this](const std::string& sensor_id, std::unique_ptr<sensor::Data> data) {
        HandleCollatedSensorData(sensor_id, std::move(data));
      });
```

综合上述的描述，数据的处理流转主要表现为：CollatedTrajectoryBuilder::AddSensorData(…)->CollatedTrajectoryBuilder::AddData(…)->sensor_collator_->AddSensorData(…)->OrderedMultiQueue::Add()->OrderedMultiQueue::Dispatch()->HandleCollatedSensorData()->data->AddToTrajectoryBuilder(wrapped_trajectory_builder_.get())->trajectory_builder->AddSensorData(sensor_id_, data_);->local_trajectory_builder_->AddRangeData()。

当数据流转到LocalTrajectoryBuilderND的AddRangeData函数时，就基本到达了SLAM算法的核心部分，包括外推器的姿态估计，ScanMatch的局部位姿优化(Local SLAM)，以及全局位姿优化(PoseGraph )等过程中的Lidar数据的进一步的流转和变换，限于篇幅，将在下一篇文章中向大家介绍。

References

[1]、cartographer中的传感器数据及代码模块介绍：./cartographer/sensors/传感器数据模块介绍.md
[2]、cartographer文档: Cartographer — Cartographer documentation