
- 关于原先`sensor/configuration`下的代码 最新代码下删除了所有关于Configuration的文件,转而被重构和分散到了不同的模块中,如下


## 从 Lua 配置文件中解析传感器配置

`cartographer\sensor\internal\voxel_filter.cc`

```c++
proto::AdaptiveVoxelFilterOptions CreateAdaptiveVoxelFilterOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::AdaptiveVoxelFilterOptions options;
  options.set_max_length(parameter_dictionary->GetDouble("max_length"));
  options.set_min_num_points(
      parameter_dictionary->GetNonNegativeInt("min_num_points"));
  options.set_max_range(parameter_dictionary->GetDouble("max_range"));
  return options;
}
```

以及`cartographer/mapping/trajectory_builder_interface.cc`下

```c++
proto::TrajectoryBuilderOptions CreateTrajectoryBuilderOptions(
    common::LuaParameterDictionary* const parameter_dictionary)

void PopulatePoseGraphOdometryMotionFilterOptions(
    proto::TrajectoryBuilderOptions* const trajectory_builder_options,
    common::LuaParameterDictionary* const parameter_dictionary) 


void PopulatePureLocalizationTrimmerOptions(
    proto::TrajectoryBuilderOptions* const trajectory_builder_options,
    common::LuaParameterDictionary* const parameter_dictionary)
```

- 配置解析职责分离到具体使用模块
- 新增传感器类型时只需修改对应模块
- 参数校验更贴近实际使用场景

## 坐标变换功能

旧版集中式转换：
```cpp
transform::Rigid3d GetTransformToTracking(
    const proto::Configuration& config, 
    const std::string& sensor_id);
```
新版分布式实现：
基础变换库 (transform/transform.cc)

```c++
Rigid3d TransformInterpolator::Interpolate(common::Time time) const {
  // 通用的插值计算逻辑
}

RangeData TransformRangeData(const RangeData& range_data,
                            const transform::Rigid3f& transform) {
  return {
    transform * range_data.origin,
    TransformPointCloud(range_data.returns, transform),
    TransformPointCloud(range_data.misses, transform)
  };
}
```

- 基础变换算法保持独立可复用
- 各传感器数据类型自主实现转换逻辑
- 支持链式转换组合（如：sensor->local->global）

## 传感器启用状态管理

轨迹构建时指定传感器白名单


`//cartographer\sensor\collator_interface.h`

```cpp
  // Adds a trajectory to produce sorted sensor output for. Calls 'callback'
  // for each collated sensor data.
  virtual void AddTrajectory(
      int trajectory_id,
      const absl::flat_hash_set<std::string>& expected_sensor_ids,
      const Callback& callback) = 0;
  // 只处理在expected_sensor_ids中的传感器
```



运行时动态过滤
```cpp
// ordered_multi_queue.cc
void AddSensorData(int trajectory_id, std::unique_ptr<Data> data) {
  if (!active_sensors_.contains(data->GetSensorId())) return;
  // 处理数据...
}
```

## 配置数据结构
配置proto文件变化：

```proto
sensor/proto/configuration.proto
message Configuration {
  repeated SensorConfig sensors = 1; 
}
```

新版结构：
```proto
sensor/proto/adaptive_voxel_filter_options.proto
message AdaptiveVoxelFilterOptions {
  double max_length = 1;
  int32 min_num_points = 2;
}
```

mapping/proto/trajectory_builder_options.proto 

```c++
message TrajectoryBuilderOptions {
  ImuTrackerOptions imu_options = 3;
  LaserScanOptions laser_scan_options = 4;
}
```





