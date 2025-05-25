上篇文章介绍到传感器消息在到达SLAM核心算法模块之前的数据预处理的方法和过程，这篇文章介绍激光点云数据在cartographer系统中统一适配处理的方法细节。

由于精心设计的代码框架，cartographer系统能较好适配多种机器人不同传感器组合。如支持兼容LaserScan 2D激光雷达(室内地面平面空间轮式机器人的配置，如仓库AGV Automated Guided Vehicle，小车自动导引运输系统小车，扫地机器人等)和PointCloud 3D激光雷达(空中无人机3D建图，室外复杂地面3维空间场景自主导航机器人，如自动泊车应用等)，而且系统自动适配机器人本体设备上多个3D Lidar传感器的点云数据的融合以支持更高精度建图(RangeDataCollator类的成员变量expected_sensor_ids_的支持，可以通过配置文件定义激光雷达的数目)。

当数据流转到LocalTrajectoryBuilderND(先以LocalTrajectoryBuilder2D类为例进行介绍)的AddRangeData函数时，就到达了SLAM算法处理的核心部分。在该函数里，主要处理逻辑有：

1、首先调用range_data_collator_.AddRangeData()函数实现一到多个Lidar传感器的点云数据的同步和融合，返回同步后的逻辑点云帧。该函数内部主要调用了CropAndMerge函数将不同来源点云传感器的数据在时间重叠区域外进行切割裁剪，在时间重叠区域内进行合并融合。裁剪融合后逻辑点云帧中部分点云点由于截至时间的变化，其相对时间信息需要做一个校正，具体的代码和文字注释如下：

```
//对每一帧(2D场景下可能是进行了剪切形成的subdivision的逻辑点云帧)不同激光传感器来源
//(不同frame_id)的点云数据，CropAndMerge函数根据计算好的裁剪融合时间区间
//[current_start_, current_end_]对点云帧的点云进行过滤，各个点云帧分别得出各自的
//起始到结束迭代器[overlap_begin, overlap_end]之间的点云点数据，将这些点云点在该时间
//跨度内进行融合操作：1、相对时间校正，部分点云帧的扫描结束时间发生了变化，需要更新这部分点
//云点的相对时间；2、数据类型转换，转换后的数据类型为TimedPointCloudOriginData。其中
//origins数组记录了不同激光传感器设备相对于tracking_frame的位移。而RangeMeasurement
//数据类型中除了记录点云点的坐标(已经转换到车体坐标系下)，点云点强度信息外，还多了一个来自哪
//个激光传感器的origin_index(origins数组的下标位置)信息。
    if (overlap_begin < overlap_end) {
      std::size_t origin_index = result.origins.size();
      result.origins.push_back(data.origin);
      const float time_correction =
          static_cast `<float>`(common::ToSeconds(data.time - current_end_));
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
        point.point_time.time += time_correction;
        result.ranges.push_back(point);
      }
    }
```

下面部分的代码实现了融合区间之外的点云帧的数据的切分，其中data为引用数据类型，直接修改了原始电源帧的数据，保留裁剪剩下的部分供和下一次融合使用。最后还对多个传感器融合后的点云数据结果根据时间进行了严格的排序。

```
// Drop buffered points until overlap_end.
if (overlap_end == ranges.end()) {
      it = id_to_pending_data_.erase(it);
    } else if (overlap_end == ranges.begin()) {
      ++it;
    } else {
      const auto intensities_overlap_end =
          intensities.begin() + (overlap_end - ranges.begin());
      data = sensor::TimedPointCloudData{
          data.time, data.origin,
          sensor::TimedPointCloud(overlap_end, ranges.end()),
          std::vector<float>(intensities_overlap_end, intensities.end())};
      ++it;
    }
```
2、在对点云数据帧进行裁剪融合同步后，对返回的逻辑点云帧的每一个点云点对应的时间启用姿态外推器进行位姿估计(关于姿态外推器的实现原理将在专门的文章中进行介绍[1])，将估计的结果存入到range_data_poses的数组里。紧接着将基于估计出的点云点时刻的位姿对点云点进行坐标变换，变换到统一的全局坐标系下的坐标以解决点云扫描时间的不同形成的点云位置畸变。统一坐标后将点云点测得的距离按range区间进行过滤。具体的代码和分析注释如下：

```
//根据每个点云点时刻的机器人本体位姿和点云点相对于机器人本体坐标系下的坐标(前面的文章[2]做过分析，
//已经将点云点在传感器坐标系下的坐标转换为了在本体坐标系下的坐标)，对点云点坐标进行去畸变变换，
//hit_in_local的结果为对齐到统一的全局坐标系下的点云点坐标，基于点云点在全局坐标系下的坐标和
//点云激光发射位置在全局坐标系的坐标origin_in_local,就可以计算出点云点距离值range(向量加减法和模运算)，
//然后基于range和系统配置的range区间范围进行过滤，将在区间内的点云数据放入returns数组，
//其他的点云数据放入到misses数组(misses可以理解为没有返回，点云发射到了空旷的空间没有回波)。
for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
    const sensor::TimedRangefinderPoint& hit =
        synchronized_data.ranges[i].point_time;
    const Eigen::Vector3f origin_in_local = range_data_poses[i] *
 synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);
    sensor::RangefinderPoint hit_in_local =
        range_data_poses[i] * sensor::ToRangefinderPoint(hit);
    const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
    const float range = delta.norm();
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit_in_local);
      } else {
        hit_in_local.position =
            origin_in_local +
            options_.missing_data_ray_length() / range * delta;
        accumulated_range_data_.misses.push_back(hit_in_local);
      }
    }
  }
```

在对当前逻辑点云帧进行了数据处理后，会将状态变量num_accumulated_进行递增，当满足条件num_accumulated_ >= options_.num_accumulated_range_data()时，就将多个逻辑点云帧的数据进行聚合并调用局部位姿优化得出局部位姿优化结果。具体的形成聚合数据的方法参见下面的代码和分析。
```
//time为多个逻辑点云帧中的最后一个点云帧的点云点最晚扫描点时间
//同时将rangedata中的原点位置定义为最后一个点云点位姿中的位移坐标值
//并将rangedata中的点云点根据函数TransformToGravityAlignedFrameAndFilter   
//实现变换，其中坐标变换transform_to_gravity_aligned_frame定义为：
//gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
//关于该公式蕴含的几何变换含义的说明请参考后面专门段落的说明。
if (num_accumulated_ >= options_.num_accumulated_range_data()) {
  const common::Time current_sensor_time = synchronized_data.time;
  absl::optional<common::Duration> sensor_duration;
  if (last_sensor_time_.has_value()) {
    sensor_duration = current_sensor_time - last_sensor_time_.value();
  }
  last_sensor_time_ = current_sensor_time;
  num_accumulated_ = 0;
  const transform::Rigid3d gravity_alignment = transform::Rigid3d::Rotation(
      extrapolator_->EstimateGravityOrientation(time));
  // TODO(gaschler): This assumes that 'range_data_poses.back()' is at time
  // 'time'.
  accumulated_range_data_.origin = range_data_poses.back().translation();
  return AddAccumulatedRangeData(
     time,
     TransformToGravityAlignedFrameAndFilter(
       gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
       accumulated_range_data_),
     gravity_alignment, sensor_duration);
  }
```

这里对TransformToGravityAlignedFrameAndFilter变换做一下具体说明。在进行变换之前，所有点云点accumulated_range_data_的坐标已经变换到全局的坐标系下(通过点云点时刻time车体相对于起始世界坐标的位姿对点云点相对于车体坐标系下的坐标点进行坐标变换实现，既点云点去畸变的实现，详见第2部分的说明)。这里的变换为两个变换的组合变换，关于这两个变换分别说明如下。

1、range_data_poses.back().inverse()为最后一个点云点对应的车体位姿的逆变换，其作用于当前逻辑点云帧的所有点云点上，将最后一个点云点的坐标定义坐标原点且坐标轴为标准方向，基于此坐标系定义，将点云点坐标进行变换到局部的坐标系下。range_data_poses基于ScanMatch算法获得较为精确的基础位姿，和局部时间段基于ImuTracker估计所得位姿变化量叠加实现点云点时刻的位姿估计。

2、gravity_alignment.cast<float>() 为重力对齐变换，其为ImuData数据根据陀螺仪的角速度和加速度计加速度进行积分和双重校正得以实现，返回的是time时刻预估的方向旋转四元数估计orientation_(和标准的坐标轴方向不一致，定义了车身的姿态的方向角度部分以及重力相对于垂直方向的角度偏移)。gravity_alignment的估计只用到了ImuData的数据，比range_data_poses.back()中的重力方向的估计更加的鲁棒和稳定，所以用gravity_alignment变换去和重力方向做对齐操作。

gravity_alignment.cast<float>() * range_data_poses.back().inverse()为两者变换的叠加，实现了将点云点转换到点云帧局部标准坐标系下后再根据当时的重力方向进行对齐(相当于将标准坐标z轴对齐到重力方向后点云点的坐标值的变化）。

将点云点坐标进行上述的变换后，AddAccumulatedRangeData函数将实现具体SLAM算法的核心部分，主要包括：1、基于上述变换后的点云点和时间通过调用函数ScanMatch进行局部位姿优化；2、InsertIntoSubmap函数基于点云数据对局部地图进行更新；3、返回匹配优化和更新后的地图结果的综合信息。关于位姿优化和局部地图更新的具体实现将在后续文章中加以详细介绍，欢迎关注。

References

[1]、cartographer中姿态外推器的实现原理分析： SLAM系列之cartographer系统中PoseExtrapolater实现分析
[2]、cartographer中的传感器数据统一预处理: SLAM系列之cartographer系统中多种类型传感器数据的统一预处理