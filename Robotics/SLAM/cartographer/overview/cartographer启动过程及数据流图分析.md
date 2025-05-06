# Cartographer启动过程及传感器消息在系统实现中的流转分析

## 启动流程示例：

```
wget -P ~/Downloads https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag
roslaunch cartographer_ros demo_backpack_2d.launch bag_filename:=${HOME}/Downloads/cartographer_paper_deutsches_museum.bag

wget -P ~/Downloads https://storage.googleapis.com/cartographer-public-data/bags/backpack_3d/with_intensities/b3-2016-04-05-14-14-00.bag
roslaunch cartographer_ros demo_backpack_3d.launch bag_filename:=${HOME}/Downloads/b3-2016-04-05-14-14-00.bag`

//roslaunch 会自动检查并在必要时启动 roscore
```

## 配置文件解读

目录 `.\cartographer_ros\cartographer_ros\configuration_files`下的应用配置文件列表如下：

```
assets_writer_backpack_2d_ci.lua*
assets_writer_backpack_2d.lua*
assets_writer_backpack_3d.lua*
assets_writer_ros_map.lua*
backpack_2d_localization_evaluation.lua*
backpack_2d_localization.lua*
backpack_2d.lua*
backpack_2d_server.lua*
backpack_3d_localization.lua*
backpack_3d.lua*
demo_2d.rviz*
demo_3d.rviz*
pr2.lua*
revo_lds.lua*
taurob_tracker.lua*
transform.lua*
visualize_pbstream.lua*
```

以 `.\cartographer_ros\cartographer_ros\configuration_files\backpack_3d.lua`为例子进行说明

其中 node_options, trajectory_options通过语句 `std::tie(node_options, trajectory_options) = LoadOptions(FLAGS_configuration_directory, FLAGS_configuration_basename);`进行加载
configuration_directory即为 `.\cartographer_ros\cartographer_ros\configuration_files`，configuration_basename为 `backpack_3d.lua`等

```
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map", # 定义 TF (Transform) 坐标系名称，这对于正确处理传感器数据至关重要
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = true, # 是否让 Cartographer 提供 odom -> base_link 的 TF 变换。通常如果使用了外部里程计，这里会设为 false，由外部里程计节点提供。
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0, # 如果为 0，则不使用 2D Lidar。如果大于 0，Cartographer 会期望 scan (如果为1) 或 scan_1, scan_2
  num_multi_echo_laser_scans = 0, # : 整数，指定使用的多回波 2D LaserScan 传感器的数量
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 2, #  整数，指定使用的 3D 点云 传感器 (Lidar) 的数量。如果为 0，则不使用 3D Lidar。如果大于 0，Cartographer 会期望 points2 (如果为1) 或 points2_1, points2_2 ... (如果大于1) 这些 topic 上的数据（
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 160

MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 7
POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.optimize_every_n_nodes = 320
POSE_GRAPH.constraint_builder.sampling_ratio = 0.03
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66

return options
```

## 启动流程

- ROS launch文件启动，启动 Cartographer 节点（如 cartographer_node）。

  `.\cartographer_ros\cartographer_ros\launch\demo_backpack_3d.launch`文件如下

```
<launch>
  <param name="/use_sim_time" value="true" />

  <include file="$(find cartographer_ros)/launch/backpack_3d.launch" />

  <node name="rviz" pkg="rviz" type="rviz" required="true"
      args="-d $(find cartographer_ros)/configuration_files/demo_3d.rviz" />
  <node name="playbag" pkg="rosbag" type="play"
      args="--clock $(arg bag_filename)" />
</launch>

```

  `.\cartographer_ros\cartographer_ros\launch\backpack_3d.launch`文件如下
`<launch>`
  `<param name="robot_description"     textfile="$(find cartographer_ros)/urdf/backpack_3d.urdf" />`

  `<node name="robot_state_publisher" pkg="robot_state_publisher"     type="robot_state_publisher" />`

  `<node name="cartographer_node" pkg="cartographer_ros"       type="cartographer_node" args="           -configuration_directory $(find cartographer_ros)/configuration_files           -configuration_basename backpack_3d.lua"       output="screen">`
    `<remap from="points2_1" to="horizontal_laser_3d" />`
    `<remap from="points2_2" to="vertical_laser_3d" />`
  `</node>`

  `<node name="cartographer_occupancy_grid_node" pkg="cartographer_ros"       type="cartographer_occupancy_grid_node" args="-resolution 0.05" />`
`</launch>`

- 入口main函数在路径 `.\cartographer_ros\cartographer_ros\cartographer_ros\node_main.cc`里，实现代码为

```
namespace cartographer_ros {
namespace {
void Run() {
  constexpr double kTfBufferCacheTimeInSeconds = 10.;
  tf2_ros::Buffer tf_buffer{::ros::Duration(kTfBufferCacheTimeInSeconds)};
  tf2_ros::TransformListener tf(tf_buffer);
  NodeOptions node_options;
  TrajectoryOptions trajectory_options;
  std::tie(node_options, trajectory_options) =
      LoadOptions(FLAGS_configuration_directory, FLAGS_configuration_basename);

  auto map_builder =
      cartographer::mapping::CreateMapBuilder(node_options.map_builder_options);
  Node node(node_options, std::move(map_builder), &tf_buffer,
            FLAGS_collect_metrics);
  if (!FLAGS_load_state_filename.empty()) {
    node.LoadState(FLAGS_load_state_filename, FLAGS_load_frozen_state);
  }

  if (FLAGS_start_trajectory_with_default_topics) {
    node.StartTrajectoryWithDefaultTopics(trajectory_options);
  }

  ::ros::spin();

  node.FinishAllTrajectories();
  node.RunFinalOptimization();

  if (!FLAGS_save_state_filename.empty()) {
    node.SerializeState(FLAGS_save_state_filename,
                        true /* include_unfinished_submaps */);
  }
}

}  // namespace
}  // namespace cartographer_ros

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_configuration_directory.empty())
      << "-configuration_directory is missing.";
  CHECK(!FLAGS_configuration_basename.empty())
      << "-configuration_basename is missing.";

  ::ros::init(argc, argv, "cartographer_node");
  ::ros::start();

  cartographer_ros::ScopedRosLogSink ros_log_sink;
  cartographer_ros::Run();
  ::ros::shutdown();
}
```

其中Node类(代码路径:`.\cartographer_ros\cartographer_ros\cartographer_ros\node.cc`)

```
void Node::StartTrajectoryWithDefaultTopics(const TrajectoryOptions& options) {
  absl::MutexLock lock(&mutex_);
  CHECK(ValidateTrajectoryOptions(options));
  AddTrajectory(options);
}

int Node::AddTrajectory(const TrajectoryOptions& options) {
    //cartographer::mapping::TrajectoryBuilderInterface::SensorId: 这是 Cartographer 核心库用来唯一标识一个传感器的结构体。它通常包含两部分
    //type: 传感器的类型（如 RANGE, IMU, ODOMETRY, LANDMARK 等枚举值）。
    //id: 一个字符串标识符，通常对应于配置文件中或代码中定义的传感器名称（如 "scan", "points2", "imu"）。
  const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
      expected_sensor_ids = ComputeExpectedSensorIds(options);//辅助函数会解析 options 结构体中的配置项
  //在 Cartographer 核心库 (libcartographer) 中实际创建和初始化一个新的轨迹处理实例 (Trajectory Builder)，并获取该轨迹的唯一ID
  const int trajectory_id =
      map_builder_bridge_.AddTrajectory(expected_sensor_ids, options); //map_builder_bridge_为连接 ROS 接口 (Node 类) 和 Cartographer 核心库 (cartographer::mapping::MapBuilder) 之间的桥梁。它负责将 ROS 消息转换为 Cartographer 内部格式，并调用核心库的函数。
  AddExtrapolator(trajectory_id, options); 
  AddSensorSamplers(trajectory_id, options);
  LaunchSubscribers(options, trajectory_id);
  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(kTopicMismatchCheckDelaySec),
      &Node::MaybeWarnAboutTopicMismatch, this, /*oneshot=*/true));
  for (const auto& sensor_id : expected_sensor_ids) {
    subscribed_topics_.insert(sensor_id.id);
  }
  return trajectory_id;
}


//以下为LaunchSubscribers函数实现中的片段，为注册点云信息的句柄
for (const std::string& topic :
    ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
subscribers_[trajectory_id].push_back(
    {SubscribeWithHandler<sensor_msgs::PointCloud2>(
            &Node::HandlePointCloud2Message, trajectory_id, topic,
            &node_handle_, this),
        topic});
}


```

通过订阅了特定的消息，并注册了相关的处理句柄后，当有消息到来，句柄会被调用，以PointCloud2为例子进行说明

```
.\cartographer_ros\cartographer_ros\cartographer_ros\sensor_bridge.cc

void SensorBridge::HandlePointCloud2Message(
    const std::string& sensor_id,
    const sensor_msgs::PointCloud2::ConstPtr& msg) {
  carto::sensor::PointCloudWithIntensities point_cloud;
  carto::common::Time time;
  std::tie(point_cloud, time) = ToPointCloudWithIntensities(*msg);
  HandleRangefinder(sensor_id, time, msg->header.frame_id, point_cloud.points);
}

void SensorBridge::HandleRangefinder(
    const std::string& sensor_id, const carto::common::Time time,
    const std::string& frame_id, const carto::sensor::TimedPointCloud& ranges) {
  if (!ranges.empty()) {
    CHECK_LE(ranges.back().time, 0.f);
  }
  const auto sensor_to_tracking =
      tf_bridge_.LookupToTracking(time, CheckNoLeadingSlash(frame_id));
  if (sensor_to_tracking != nullptr) {
    if (IgnoreMessage(sensor_id, time)) {
      LOG(WARNING) << "Ignored Rangefinder message from sensor " << sensor_id
                   << " because sensor time " << time
                   << " is not before last Rangefinder message time "
                   << latest_sensor_time_[sensor_id];
      return;
    }
    latest_sensor_time_[sensor_id] = time;
    trajectory_builder_->AddSensorData(
        sensor_id, carto::sensor::TimedPointCloudData{
                       time, sensor_to_tracking->translation().cast<float>(),
                       carto::sensor::TransformTimedPointCloud(
                           ranges, sensor_to_tracking->cast<float>())});
  }
}

//.\cartographer\mapping\internal\collated_trajectory_builder.h
void AddSensorData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& timed_point_cloud_data) override {
  AddData(sensor::MakeDispatchable(sensor_id, timed_point_cloud_data));
}

//  sensor::CollatorInterface* const sensor_collator_;
void CollatedTrajectoryBuilder::AddData(std::unique_ptr<sensor::Data> data) {
  sensor_collator_->AddSensorData(trajectory_id_, std::move(data));
}

//Cartographer 通常会处理来自多个传感器（例如，多个激光雷达、IMU、里程计、GPS、固定帧位姿等）的数据，
//并且这些传感器数据以不同的频率和时间戳到达系统。
//为了保证 SLAM 算法能够处理时间同步的数据集（即，在某个时间点附近的所有相关传感器数据），
//需要一个机制来收集、排序和分发这些数据。这就是 Sensor Collator 的作用。

if (options.collate_by_trajectory()) {
  //这是更复杂的版本，专门设计用来同时处理来自多条不同轨迹的传感器数据。
  sensor_collator_ = absl::make_unique<sensor::TrajectoryCollator>();
} else {
  //这是简单版本的整理器。它假设系统只处理一条轨迹
  sensor_collator_ = absl::make_unique<sensor::Collator>();
}



void TrajectoryCollator::AddSensorData(const int trajectory_id,
                                       std::unique_ptr<Data> data) {
  QueueKey queue_key{trajectory_id, data->GetSensorId()};
  auto* metric = GetOrCreateSensorMetric(data->GetSensorId(), trajectory_id);
  metric->Increment();
  trajectory_to_queue_.at(trajectory_id)
      .Add(std::move(queue_key), std::move(data));
}

void OrderedMultiQueue::Add(const QueueKey& queue_key,
                            std::unique_ptr<Data> data) {
  auto it = queues_.find(queue_key);
  if (it == queues_.end()) {
    LOG_EVERY_N(WARNING, 1000)
        << "Ignored data for queue: '" << queue_key << "'";
    return;
  }
  it->second.queue.Push(std::move(data));
  Dispatch();
}

void TrajectoryCollator::AddTrajectory(
    const int trajectory_id,
    const absl::flat_hash_set<std::string>& expected_sensor_ids,
    const Callback& callback) {
  CHECK_EQ(trajectory_to_queue_.count(trajectory_id), 0);
  for (const auto& sensor_id : expected_sensor_ids) {
    const auto queue_key = QueueKey{trajectory_id, sensor_id};
    trajectory_to_queue_[trajectory_id].AddQueue(
        queue_key, [callback, sensor_id](std::unique_ptr<Data> data) {
          callback(sensor_id, std::move(data));
        });
    trajectory_to_queue_keys_[trajectory_id].push_back(queue_key);
  }
}

//CollatedTrajectoryBuilder构造函数里
sensor_collator_->AddTrajectory(
    trajectory_id, expected_sensor_id_strings,
    [this](const std::string& sensor_id, std::unique_ptr<sensor::Data> data) {
      HandleCollatedSensorData(sensor_id, std::move(data));
    });


//HandleCollatedSensorData
data->AddToTrajectoryBuilder(wrapped_trajectory_builder_.get());

void AddToTrajectoryBuilder(
    mapping::TrajectoryBuilderInterface *const trajectory_builder) override {
  trajectory_builder->AddSensorData(sensor_id_, data_);
}

//GlobalTrajectoryBuilder
void AddSensorData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& timed_point_cloud_data) override {
  CHECK(local_trajectory_builder_)
      << "Cannot add TimedPointCloudData without a LocalTrajectoryBuilder.";
  std::unique_ptr<typename LocalTrajectoryBuilder::MatchingResult>
      matching_result = local_trajectory_builder_->AddRangeData(
          sensor_id, timed_point_cloud_data);
  if (matching_result == nullptr) {
    // The range data has not been fully accumulated yet.
    return;
  }


node-》sensor_bridge-》
trajectory_builder_->AddSensorData（collated_trajectory_builder,add to queue）-》Dispatch-》
HandleCollatedSensorData(data->AddToTrajectoryBuilder(wrapped_trajectory_builder_.get());)-》
trajectory_builder->AddSensorData(sensor_id_, data_);-》//GlobalTrajectoryBuilder
local_trajectory_builder_->AddRangeData（GlobalTrajectoryBuilder的AddSensorData函数）//local slam进行处理
```

### 问题1：2D Lidar（LaserScan）点云数据帧为什么要进行切分？切分在哪里进行？切分的逻辑如何？

在SensorBridge类里，根据配置文件里设置的
```
num_multi_echo_laser_scans = 1,
num_subdivisions_per_laser_scan = 10,
```
其中参数`num_subdivisions_per_laser_scan`只针对2D Lidar数据有效，当机器人运动时，单次激光扫描（如 360° 扫描）的时间跨度内，机器人可能已发生位移或旋转，导致扫描数据存在运动畸变。直接使用整帧扫描进行匹配会产生误差。

2D LiDAR扫描点的时序性与角度严格对应，这种切分等价于按角度切分
点云数组（ranges）按扫描顺序存储，数组索引直接对应角度顺序。例如：
ranges[0]对应0°（起始角度），ranges[359]对应359.75°。

```
void SensorBridge::HandleLaserScan(
    const std::string& sensor_id, const carto::common::Time time,
    const std::string& frame_id,
    const carto::sensor::PointCloudWithIntensities& points) {
  if (points.points.empty()) {
    return;
  }
  CHECK_LE(points.points.back().time, 0.f);
  // TODO(gaschler): Use per-point time instead of subdivisions.
  for (int i = 0; i != num_subdivisions_per_laser_scan_; ++i) {
    const size_t start_index =
        points.points.size() * i / num_subdivisions_per_laser_scan_;
    const size_t end_index =
        points.points.size() * (i + 1) / num_subdivisions_per_laser_scan_;
    carto::sensor::TimedPointCloud subdivision(
        points.points.begin() + start_index, points.points.begin() + end_index);
    if (start_index == end_index) {
      continue;
    }
    const double time_to_subdivision_end = subdivision.back().time;
    // `subdivision_time` is the end of the measurement so sensor::Collator will
    // send all other sensor data first.
    const carto::common::Time subdivision_time =
        time + carto::common::FromSeconds(time_to_subdivision_end);
    auto it = sensor_to_previous_subdivision_time_.find(sensor_id);
    if (it != sensor_to_previous_subdivision_time_.end() &&
        it->second >= subdivision_time) {
      LOG(WARNING) << "Ignored subdivision of a LaserScan message from sensor "
                   << sensor_id << " because previous subdivision time "
                   << it->second << " is not before current subdivision time "
                   << subdivision_time;
      continue;
    }
    sensor_to_previous_subdivision_time_[sensor_id] = subdivision_time;
    for (auto& point : subdivision) {
      point.time -= time_to_subdivision_end;
    }
    CHECK_EQ(subdivision.back().time, 0.f);
    HandleRangefinder(sensor_id, subdivision_time, frame_id, subdivision);
  }
}
```

在3D Lidar点云数据中，一般不使用切分，主要原因有：
- 一是多束激光雷达的特性，3D点云需保持空间完整性以提取平面、曲面等特征，切分会破坏结构。
- 通过IMU或连续时间轨迹优化直接校正点云畸变，无需分段处理。
- 采用体素滤波（如Cartographer的3D模式）降低数据量，而非切分。

### 问题2：多个Lidarsensor的数据如何进行合并，都是基于subdivision进行的合并吗？




## 如何通过服务在已经启动的cartographer ros节点中添加新的trajectory用于多机器协同建图或分段建图？

```
//我这个节点提供一个服务，欢迎其他节点来请求我
//当有其他节点用 call 或 ServiceProxy 请求这个服务时，就会触发你注册的处理函数
//  ::ros::NodeHandle node_handle_;
service_servers_.push_back(node_handle_.advertiseService(
    kStartTrajectoryServiceName, &Node::HandleStartTrajectory, this));

```

在cartographer的多机器人系统中，通常使用的是 cartographer_grpc 提供的客户端机制进行启动和通信。这些客户端不是通过传统 ROS 的 topic 订阅方式，而是通过 gRPC 接口发送传感器数据和启动轨迹的。

```
roslaunch cartographer_grpc trajectory_builder_client.launch

trajectory_builder_main -configuration_directory /your/config/path \
                        -configuration_basename trajectory_builder.lua \
                        -server_address localhost:50051 \
                        -client_id "robot1"
```

在多机器人系统中，每个机器人如何共享地图信心？
主要有两个方法：

- 使用 cartographer_ros 的 cartographer_occupancy_grid_node，订阅来自 map_builder_server 的 submap 信息，生成栅格地图并发布：`/map             (nav_msgs/OccupancyGrid)`
- 通过 gRPC 调用服务器接口，获取地图信息（适合非 ROS 的客户端）
- map_builder_server 端可以通过：`rosservice call /write_state ...`,将当前地图保存成 .pbstream 文件(离线建图、恢复运行)。然后其他机器人（如果需要离线加载）可以加载这个地图文件，通过：`node.LoadState("xxx.pbstream", frozen_state=false)`
