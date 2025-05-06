# submap_3d.h

## submap_3d

submap_3d继承submap,是一个用于三维建图的类，主要用于存储和管理三维空间中的子地图,submap3D 类用于表示在三维空间中局部区域的地图数据，包括传感器的扫描数据（如激光雷达、深度相机等）的插入和处理


```c++
class Submap3D : public Submap {
 private:
  std::unique_ptr<HybridGrid> high_resolution_hybrid_grid_;  // 高分辨率混合网格
  std::unique_ptr<HybridGrid> low_resolution_hybrid_grid_;   // 低分辨率混合网格 
  std::unique_ptr<IntensityHybridGrid> high_resolution_intensity_hybrid_grid_; // 高分辨率强度网格
  Eigen::VectorXf rotational_scan_matcher_histogram_;  // 旋转扫描匹配直方图
};
```


## 核心算法


- InsertData
进行对两个网格的维护,更新旋转扫描匹配器直方图

```c++
void Submap3D::InsertData(const sensor::RangeData& range_data_in_local,
                          const RangeDataInserter3D& range_data_inserter,
                          const float high_resolution_max_range,
                          const Eigen::Quaterniond& local_from_gravity_aligned,
                          const Eigen::VectorXf& scan_histogram_in_gravity) {
  // 确保插入的数据操作没有结束（即插入没有被标记为完成）
  CHECK(!insertion_finished());

  // 将传感器数据从局部坐标系转换为子地图坐标系
  const sensor::RangeData transformed_range_data = sensor::TransformRangeData(
      range_data_in_local, local_pose().inverse().cast<float>());

  // 插入数据到高分辨率的网格中，使用 max_range 过滤数据（只保留在指定最大范围内的扫描数据）
  range_data_inserter.Insert(
      FilterRangeDataByMaxRange(transformed_range_data,
                                high_resolution_max_range),
      high_resolution_hybrid_grid_.get(),
      high_resolution_intensity_hybrid_grid_.get());

  // 插入数据到低分辨率的网格中（不包含强度数据）
  range_data_inserter.Insert(transformed_range_data,
                             low_resolution_hybrid_grid_.get(),
                             /*intensity_hybrid_grid=*/nullptr);

  // 增加已经插入的数据计数
  set_num_range_data(num_range_data() + 1);

  // 计算子地图相对于重力对齐的旋转角度（以子地图为参考坐标系）
  const float yaw_in_submap_from_gravity = transform::GetYaw(
      local_pose().inverse().rotation() * local_from_gravity_aligned);

  // 更新旋转扫描匹配器直方图，通过旋转扫描匹配器对扫描直方图进行旋转
  rotational_scan_matcher_histogram_ +=
      scan_matching::RotationalScanMatcher::RotateHistogram(
          scan_histogram_in_gravity, yaw_in_submap_from_gravity);
}
```

算法流程：
- 数据插入时维护两个活跃子图（高分辨率和低分辨率）
- 当新子图达到一定数据量后，将其标记为完成并创建新的子图
- 更新旋转扫描匹配器直方图，通过旋转扫描匹配器对扫描直方图进行旋转

## ActiveSubmaps3D

ActiveSubmaps3D 是管理多个 Submap3D 实例的类

它通过使用vector来进行对Submap3D的管理

```c++
class ActiveSubmaps3D {
 public:

...
 private:
  void AddSubmap(const transform::Rigid3d& local_submap_pose,
                 int rotational_scan_matcher_histogram_size);

  const proto::SubmapsOptions3D options_;
  std::vector<std::shared_ptr<Submap3D>> submaps_;
  RangeDataInserter3D range_data_inserter_;
  ```


### 核心算法

- InsertData

对管理的所有submap进行插入数据

```c++
std::vector<std::shared_ptr<const Submap3D>> ActiveSubmaps3D::InsertData(
    const sensor::RangeData& range_data,
    const Eigen::Quaterniond& local_from_gravity_aligned,
    const Eigen::VectorXf& rotational_scan_matcher_histogram_in_gravity) {
  // 如果没有子地图或最后一个子地图已满，创建新子地图
  if (submaps_.empty() ||
      submaps_.back()->num_range_data() == options_.num_range_data()) {
    AddSubmap(transform::Rigid3d(range_data.origin.cast<double>(),
                               local_from_gravity_aligned),
            rotational_scan_matcher_histogram_in_gravity.size());
  }
  
  // 将数据插入所有活动子地图
  for (auto& submap : submaps_) {
    submap->InsertData(range_data, range_data_inserter_,
                     options_.high_resolution_max_range(),
                     local_from_gravity_aligned,
                     rotational_scan_matcher_histogram_in_gravity);
  }
  
  // 如果第一个子地图达到阈值，标记为完成
  if (submaps_.front()->num_range_data() == 2 * options_.num_range_data()) {
    submaps_.front()->Finish();
  }
  
  return submaps();
}
```

- AddSubmap

负责管理子地图的创建和内存优化
```c++
void ActiveSubmaps3D::AddSubmap(
    const transform::Rigid3d& local_submap_pose,
    const int rotational_scan_matcher_histogram_size) {
  if (submaps_.size() >= 2) {
    // 这里会在插入新子图前裁剪已完成的子图，以减少峰值内存使用
    CHECK(submaps_.front()->insertion_finished());
    // 我们使用`ForgetIntensityHybridGrid`来减少内存使用。由于我们使用
    // 活动子图及其关联的强度混合网格进行扫描匹配，一旦我们从活动子图中
    // 移除子图并且不再需要强度混合网格，就调用`ForgetIntensityHybridGrid`
    submaps_.front()->ForgetIntensityHybridGrid();
    submaps_.erase(submaps_.begin());
  }
  // 创建一个全零初始化的旋转扫描匹配直方图
  const Eigen::VectorXf initial_rotational_scan_matcher_histogram =
      Eigen::VectorXf::Zero(rotational_scan_matcher_histogram_size);
  // 创建并添加新的子图到活动子图集合中
  submaps_.emplace_back(new Submap3D(
      options_.high_resolution(), options_.low_resolution(), local_submap_pose,
      initial_rotational_scan_matcher_histogram));
}
```