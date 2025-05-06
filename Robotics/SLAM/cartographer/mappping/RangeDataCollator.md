# RangeDataCollator功能分析

## 基本处理逻辑分析

- 传感器的数据类型有

```
SensorType::RANGE//laserscan和PointCloud统一称为range
SensorType::IMU//惯性测量单元
SensorType::ODOMETRY//里程计
SensorType::FIXED_FRAME_POSE//
SensorType::LANDMARK//路标，如二维码，标记牌等
```
- 首先在ROS节点端读取配置文件，`ComputeExpectedSensorIds`函数解析出需要哪些sensor的topic数据(放入到集合里)

```
std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
Node::ComputeExpectedSensorIds(const TrajectoryOptions& options) const {
  using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;
  using SensorType = SensorId::SensorType;
  std::set<SensorId> expected_topics;
  // Subscribe to all laser scan, multi echo laser scan, and point cloud topics.
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kLaserScanTopic, options.num_laser_scans)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  for (const std::string& topic : ComputeRepeatedTopicNames(
           kMultiEchoLaserScanTopic, options.num_multi_echo_laser_scans)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  for (const std::string& topic :
       ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
    expected_topics.insert(SensorId{SensorType::RANGE, topic});
  }
  // For 2D SLAM, subscribe to the IMU if we expect it. For 3D SLAM, the IMU is
  // required.
  if (node_options_.map_builder_options.use_trajectory_builder_3d() ||
      (node_options_.map_builder_options.use_trajectory_builder_2d() &&
       options.trajectory_builder_options.trajectory_builder_2d_options()
           .use_imu_data())) {
    expected_topics.insert(SensorId{SensorType::IMU, kImuTopic});
  }
  // Odometry is optional.
  if (options.use_odometry) {
    expected_topics.insert(SensorId{SensorType::ODOMETRY, kOdometryTopic});
  }
  // NavSatFix is optional.
  if (options.use_nav_sat) {
    expected_topics.insert(
        SensorId{SensorType::FIXED_FRAME_POSE, kNavSatFixTopic});
  }
  // Landmark is optional.
  if (options.use_landmarks) {
    expected_topics.insert(SensorId{SensorType::LANDMARK, kLandmarkTopic});
  }
  return expected_topics;
}
```
- 在LocalTrajectoryBuilder2D的实例生成时，会根据所有的传感器信息通过`SelectRangeSensorIds`函数生成点云传感器相关的信息

```
std::unique_ptr<LocalTrajectoryBuilder2D> local_trajectory_builder;
if (trajectory_options.has_trajectory_builder_2d_options()) {
    local_trajectory_builder = absl::make_unique<LocalTrajectoryBuilder2D>(
        trajectory_options.trajectory_builder_2d_options(),
        SelectRangeSensorIds(expected_sensor_ids));
}

std::vector<std::string> SelectRangeSensorIds(
    const std::set<MapBuilder::SensorId>& expected_sensor_ids) {
  std::vector<std::string> range_sensor_ids;
  for (const MapBuilder::SensorId& sensor_id : expected_sensor_ids) {
    if (sensor_id.type == MapBuilder::SensorId::SensorType::RANGE) {
      range_sensor_ids.push_back(sensor_id.id);
    }
  }
  return range_sensor_ids;
}
```
- `RangeDataCollator`类的`AddRangeData`函数实现，将不同传感器（如多个 LiDAR、雷达等）的数据对齐到统一的时间窗口内，合并为单一数据流。

几个关键的数据结构定义：

sensor::TimedPointCloudData& data

```
struct TimedPointCloudData {
  common::Time time; //点云数据的基准时间戳（传感器扫描起始时间）
  Eigen::Vector3f origin; //传感器坐标系原点（相对于机器人基坐标系）
  TimedPointCloud ranges; //由多个点组成的数组，每个点包含相对于 time 的时间偏移（point.time 字段，单位为秒）
  // 'intensities' has to be same size as 'ranges', or empty.
  std::vector<float> intensities; //强度信息（与ranges一一对应）
};
```

```
struct TimedPointCloudOriginData {
  struct RangeMeasurement {
    TimedRangefinderPoint point_time;
    float intensity;
    size_t origin_index;
  };
  common::Time time;
  std::vector<Eigen::Vector3f> origins;
  std::vector<RangeMeasurement> ranges;
};
```

```
struct TimedRangefinderPoint {
  Eigen::Vector3f position;
  float time;
};

```

```
//.\cartographer_ros\cartographer_ros\cartographer_ros\msg_conversion.cc

// For sensor_msgs::LaserScan and sensor_msgs::MultiEchoLaserScan.
template <typename LaserMessageType>
std::tuple<PointCloudWithIntensities, ::cartographer::common::Time>
LaserScanToPointCloudWithIntensities(const LaserMessageType& msg) {
  CHECK_GE(msg.range_min, 0.f);//确保最小距离非负
  CHECK_GE(msg.range_max, msg.range_min);// 确保最大距离≥最小距离
  if (msg.angle_increment > 0.f) {
    CHECK_GT(msg.angle_max, msg.angle_min);// 角度递增时，终止角需>起始角
  } else {
    CHECK_GT(msg.angle_min, msg.angle_max);// 角度递减时，起始角需>终止角
  }
  PointCloudWithIntensities point_cloud;// 存储转换后的点云和强度
  float angle = msg.angle_min; // 起始扫描角度
  for (size_t i = 0; i < msg.ranges.size(); ++i) {
    const auto& echoes = msg.ranges[i];// 检查是否存在有效回波
    if (HasEcho(echoes)) {
      const float first_echo = GetFirstEcho(echoes);// 取第一个回波的距离
      if (msg.range_min <= first_echo && first_echo <= msg.range_max) {
        const Eigen::AngleAxisf rotation(angle, Eigen::Vector3f::UnitZ());
        const cartographer::sensor::TimedRangefinderPoint point{
            rotation * (first_echo * Eigen::Vector3f::UnitX()),
            i * msg.time_increment};//时间戳=点索引×时间增量
        point_cloud.points.push_back(point);
        if (msg.intensities.size() > 0) {
          CHECK_EQ(msg.intensities.size(), msg.ranges.size());
          const auto& echo_intensities = msg.intensities[i];
          CHECK(HasEcho(echo_intensities));
          point_cloud.intensities.push_back(GetFirstEcho(echo_intensities));
        } else {
          point_cloud.intensities.push_back(0.f);
        }
      }
    }
    angle += msg.angle_increment;
  }
  ::cartographer::common::Time timestamp = FromRos(msg.header.stamp);
  if (!point_cloud.points.empty()) {
    const double duration = point_cloud.points.back().time;// 最后一点的时间偏移=扫描总时长
    timestamp += cartographer::common::FromSeconds(duration);// 修正为扫描结束时间
    for (auto& point : point_cloud.points) {
      //调整所有点的时间为相对于扫描结束的负偏移
      point.time -= duration;
      //例如：总时长0.1s → 点时间变为[-0.1, 0]
    }
  }
  //timestamp返回的是扫描帧（切片）的结束时间
  return std::make_tuple(point_cloud, timestamp);
  );
}

```

对于PointCloud2点云数据，许多（尤其是旋转式）激光雷达的驱动程序在生成 sensor_msgs::PointCloud2 消息时，会按照采集顺序填充点云数据。对于一次完整的扫描（例如 360 度旋转），最后一个被测量到的点自然具有相对于扫描开始的最大时间偏移。驱动程序会将这个相对时间戳（通常相对于消息头的时间戳 msg.header.stamp）写入每个点的 'time' 字段（如果配置了的话）。

```
sensor::TimedPointCloudOriginData RangeDataCollator::AddRangeData(
    const std::string& sensor_id,
    sensor::TimedPointCloudData timed_point_cloud_data) {
  //在配置文件里会定义哪些sensor
  CHECK_NE(expected_sensor_ids_.count(sensor_id), 0);
  timed_point_cloud_data.intensities.resize(
      timed_point_cloud_data.ranges.size(), kDefaultIntensityValue);
  // TODO(gaschler): These two cases can probably be one.
  // id_to_pending_data_ 存储各传感器待处理的数据队列，键为传感器 ID，值为该传感器未处理的点云数据。
  // 定义当前处理的时间窗口 [current_start_, current_end_]，所有传感器的数据需在此窗口内对齐。
  if (id_to_pending_data_.count(sensor_id) != 0) {
    //Case A：同一传感器新数据到达，触发旧数据处理
    current_start_ = current_end_;
    // When we have two messages of the same sensor, move forward the older of
    // the two (do not send out current).
    current_end_ = id_to_pending_data_.at(sensor_id).time;//旧数据时间戳为新窗口结束
    auto result = CropAndMerge();//合并旧数据
    id_to_pending_data_.emplace(sensor_id, std::move(timed_point_cloud_data));// 存入新数据
    return result;
  }
  //新传感器数据到达，暂存数据
  id_to_pending_data_.emplace(sensor_id, std::move(timed_point_cloud_data));
  //检查是否所有传感器数据到齐？
  if (expected_sensor_ids_.size() != id_to_pending_data_.size()) {
    //未到齐，暂不处理
    return {};
  }
  current_start_ = current_end_;
  // We have messages from all sensors, move forward to oldest.
  common::Time oldest_timestamp = common::Time::max();
  // 遍历所有数据，取最小时间戳
  for (const auto& pair : id_to_pending_data_) {
    oldest_timestamp = std::min(oldest_timestamp, pair.second.time);
  }
  current_end_ = oldest_timestamp;
  return CropAndMerge();
}
```
- CropAndMerge函数实现了将所有的点云数据源按照区间[current_start_, current_end_]进行裁剪，并

```
sensor::TimedPointCloudOriginData RangeDataCollator::CropAndMerge() {
  //1. 初始化结果对象，时间戳为current_end_
  //以current_end_为基准
  sensor::TimedPointCloudOriginData result{current_end_, {}, {}};
  bool warned_for_dropped_points = false;
  for (auto it = id_to_pending_data_.begin();
       it != id_to_pending_data_.end();) {
    sensor::TimedPointCloudData& data = it->second;
    const sensor::TimedPointCloud& ranges = it->second.ranges;
    const std::vector<float>& intensities = it->second.intensities;
    //以下两个循环的逻辑为找到当前传感器数据在时间窗口[current_start_, current_end_]
    //之间的数据点云时间跨度[overlap_begin, overlap_end]
    auto overlap_begin = ranges.begin();
    while (overlap_begin < ranges.end() &&
           data.time + common::FromSeconds((*overlap_begin).time) <
               current_start_) {
      ++overlap_begin;
    }
    auto overlap_end = overlap_begin;
    while (overlap_end < ranges.end() &&
           data.time + common::FromSeconds((*overlap_end).time) <=
               current_end_) {
      ++overlap_end;
    }
    if (ranges.begin() < overlap_begin && !warned_for_dropped_points) {
      LOG(WARNING) << "Dropped " << std::distance(ranges.begin(), overlap_begin)
                   << " earlier points.";
      warned_for_dropped_points = true;
    }

    // Copy overlapping range.
    if (overlap_begin < overlap_end) {
      std::size_t origin_index = result.origins.size();
      result.origins.push_back(data.origin);
      const float time_correction =
          static_cast<float>(common::ToSeconds(data.time - current_end_));
      auto intensities_overlap_it =
          intensities.begin() + (overlap_begin - ranges.begin());
      result.ranges.reserve(result.ranges.size() +
                            std::distance(overlap_begin, overlap_end));
      for (auto overlap_it = overlap_begin; overlap_it != overlap_end;
           ++overlap_it, ++intensities_overlap_it) {
        sensor::TimedPointCloudOriginData::RangeMeasurement point{
            *overlap_it, *intensities_overlap_it, origin_index};
        // current_end_ + point_time[3]_after == in_timestamp +
        // point_time[3]_before
        //将点的时间从 传感器本地时间（以 data.time 为基准）转换为 全局时间（以 current_end_ 为基准）。
        point.point_time.time += time_correction;
        result.ranges.push_back(point);
      }
    }

    // Drop buffered points until overlap_end.
    if (overlap_end == ranges.end()) {
      it = id_to_pending_data_.erase(it);
    } else if (overlap_end == ranges.begin()) {
      ++it;
    } else {
      const auto intensities_overlap_end =
          intensities.begin() + (overlap_end - ranges.begin());
      //剩下的窗口之后的数据还是保留到it对应的传感器数据里供下一次来进行使用
      //并非未使用，而是通过引用直接修改了容器中的待处理数据。
      //data是引用类型
      data = sensor::TimedPointCloudData{
          data.time, data.origin,
          sensor::TimedPointCloud(overlap_end, ranges.end()),
          std::vector<float>(intensities_overlap_end, intensities.end())};
      ++it;
    }
  }
  //按点的时间进行排序
  std::sort(result.ranges.begin(), result.ranges.end(),
            [](const sensor::TimedPointCloudOriginData::RangeMeasurement& a,
               const sensor::TimedPointCloudOriginData::RangeMeasurement& b) {
              return a.point_time.time < b.point_time.time;
            });
  return result;
}
```

## 相关问题
- 是否对多个PointCloud2点云数据做cropandmerge？
单个点云设备扫描的结果，如果有多于两个3d lidar点云设备，其时间戳信息是不是不完全重叠？
是将多个点云帧数据进行cropandmerge呢还是单个点云帧单独处理，也就是rangedatacollator不对不同的点云数据进行融合处理？
- 合并的时候是否是对laserscan的多个切片进行合并？如何要对切片进行合并，如何知晓切片属于一帧？
是根据配置中的切片的数目定义的吗？
