# SLAM系列之cartographer系统中的ScanMatch 2D算法分析

这篇文章将向大家介绍cartographer系统中2D SLAM中的扫描匹配算法，该算法基于姿态外推器的车体位姿估计初始值和点云帧信息实现了位姿估计的进一步优化。
下面将对其实现代码和注释做具体的说明分析。

首先是ScanMatch算法调用前的一些预处理，其代码和注释说明如下。
```
//.\cartographer\mapping\internal\2d\local_trajectory_builder_2d.cc
//Computes a gravity aligned pose prediction.
//non_gravity_aligned_pose_prediction为姿态外推器预估姿态。没有充分考虑重力对齐的位姿预测
//ExtrapolatePose函数返回的位姿没有充分考虑和重力对齐,同系列相关文章做过一些分析说明
const transform::Rigid3d non_gravity_aligned_pose_prediction =
      extrapolator_->ExtrapolatePose(time);
//在2D SLAM场景，只考虑平面位姿的优化。将姿态外推器预估姿态和更加鲁棒的重力方向旋转估计
//gravity_alignment对齐后投影变换到二维平面位姿pose_prediction(x,y位移和朝向角yaw)
//gravity_alignment是一个旋转估计(Rigid3d 中的 Eigen::Quaterniond四元数来表示)
const transform::Rigid2d pose_prediction = transform::Project2D(
   non_gravity_aligned_pose_prediction * gravity_alignment.inverse());
//gravity_aligned_range_data为前文已经介绍过的做过和重力对齐的坐标变换的点云数据，
//其坐标原点和全局坐标系原点一致，自适应体素滤波器的作用为通过对点云点数量进行降采样减
//少点云点数量的同时保留点云重要几何结构。起到加快扫描匹配位姿优化算法效率的作用
const sensor::PointCloud& filtered_gravity_aligned_point_cloud =
      sensor::AdaptiveVoxelFilter(gravity_aligned_range_data.returns,
           options_.adaptive_voxel_filter_options());
  if (filtered_gravity_aligned_point_cloud.empty()) {
    return nullptr;
  }
// local map frame <- gravity-aligned frame，调用扫描匹配优化算法进一步优化位姿
std::unique_ptr<transform::Rigid2d> pose_estimate_2d =
ScanMatch(time, pose_prediction, filtered_gravity_aligned_point_cloud);
if (pose_estimate_2d == nullptr) {
   LOG(WARNING) << "Scan matching failed.";
   return nullptr;
 }
//将优化后的位姿嵌入到3D位姿并和重力向量对齐后更新到姿态外推器
//作为点云帧时间边界的位姿优化结果放入位姿队列中。
 const transform::Rigid3d pose_estimate =
     transform::Embed3D(*pose_estimate_2d) * gravity_alignment;
 extrapolator_->AddPose(time, pose_estimate);
```
ScanMatch算法的流程代码及注释说明如下：
```
//.\cartographer\mapping\internal\2d\local_trajectory_builder_2d.cc
//active_submaps_.submaps()为std::vector数组容器，front为容器的第一个元素
//更早插入容器的元素的子地图作为局部位姿优化的参考地图，被选中的子图本身包含了多帧
//点云及其位姿信息，由于活动子图都会包含当前点云帧数据，子图之间有部分数据存在重叠
//同时不用更多的子图信息一定程度上也加快了优化效率。
std::unique_ptr<transform::Rigid2d> LocalTrajectoryBuilder2D::ScanMatch(
    const common::Time time, const transform::Rigid2d& pose_prediction,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud) {
  if (active_submaps_.submaps().empty()) {
    return absl::make_unique<transform::Rigid2d>(pose_prediction);
  }
  std::shared_ptr<const Submap2D> matching_submap =
      active_submaps_.submaps().front();
//The online correlative scan matcher will refine the initial estimate for
//the Ceres scan matcher.
  transform::Rigid2d initial_ceres_pose = pose_prediction;
//如果配置了在线的相关性扫描匹配算法，则执行实时相关性扫描算法提供更好的位姿初始值估计
//在位姿初始估计的误差较大时较为有用；
if (options_.use_online_correlative_scan_matching()) {
    const double score = real_time_correlative_scan_matcher_.Match(
        pose_prediction, filtered_gravity_aligned_point_cloud,
        *matching_submap->grid(), &initial_ceres_pose);
    kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
  }
//采用ceres非线性优化实现更加精确的位姿估计
auto pose_observation = absl::make_unique<transform::Rigid2d>();
  ceres::Solver::Summary summary;
  ceres_scan_matcher_.Match(pose_prediction.translation(),initial_ceres_pose, 
                 filtered_gravity_aligned_point_cloud,
                 *matching_submap->grid(), pose_observation.get(),&summary);
//结果更新到观测的指标实现算法过程的性能monitor
if (pose_observation) {
    kCeresScanMatcherCostMetric->Observe(summary.final_cost);
  	//.....
  }
  return pose_observation;
}
```
关于ceres的非线性优化方法的一点说明，其采用了自定义的损失函数和正则化的约束损失函数相结合的损失函数定义，自定义的损失函数的总体思路是对点云点基于位姿做坐标变换后和较为准确的地图数据做对照计算，如点云点总体上处理在概率值较大的或TSDF值接近0的栅格位置附近，则损失函数较小，否则偏移越大损失越大，通过调整位姿使得损失函数较小时的位姿为更加和地图匹配和对齐的位姿，具体实现上通过位姿损失函数和正则化约束函数采用ceres非线性优化库实现对位姿的优化求解。代码片段示意如下：

```
//./cartographer\mapping\internal\2d\scan_matching\ceres_scan_matcher_2d.cc
//基于ceres的优化问题定义和求解实现流程示意
void CeresScanMatcher2D::Match(const Eigen::Vector2d& target_translation,
                               const transform::Rigid2d& initial_pose_estimate,
                               const sensor::PointCloud& point_cloud,
                               const Grid2D& grid,
                               transform::Rigid2d* const pose_estimate,
                               ceres::Solver::Summary* const summary) const {
  double ceres_pose_estimate[3] = {initial_pose_estimate.translation().x(),
                                   initial_pose_estimate.translation().y(),
                                   initial_pose_estimate.rotation().angle()};
  ceres::Problem problem;
  CHECK_GT(options_.occupied_space_weight(), 0.);
  switch (grid.GetGridType()) {
    case GridType::PROBABILITY_GRID:
      problem.AddResidualBlock(
          CreateOccupiedSpaceCostFunction2D(
              options_.occupied_space_weight() /
                  std::sqrt(static_cast<double>(point_cloud.size())),
              point_cloud, grid),
          nullptr /* loss function */, ceres_pose_estimate);
      break;
    case GridType::TSDF:
      //....
  }
  //add translation regularization loss
  CHECK_GT(options_.translation_weight(), 0.);
  problem.AddResidualBlock(
      TranslationDeltaCostFunctor2D::CreateAutoDiffCostFunction(
          options_.translation_weight(), target_translation),
      nullptr /* loss function */, ceres_pose_estimate);
  //add rotation regularization loss ...

  ceres::Solve(ceres_solver_options_, &problem, summary);

  *pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);
}
```

这里代码和注释其中的基于占用概率的loss function定义的部分关键代码片段。

```
//.\cartographer\mapping\internal\2d\scan_matching\occupied_space_cost_function_2d.cc
//typical structure for a Ceres Solver Cost Functor
//operator()函数在ceres迭代优化过程中不断的调用去计算每一个点云点的残差，该残差衡量了点云点
//和地图的不匹配程度，或者说点云点在当前位姿下与地图对齐的程度，因为地图是在前面建图的过程中位姿优化
//出的结果，因此地图可以认为是精确度较高的，基于地图和点云点调整当前的位姿，使得点云和地图对齐度较高
//的位姿为优化的结果。
template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const GridArrayAdapter adapter(grid_);
    ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);
    const MapLimits& limits = grid_.limits();

    for (size_t i = 0; i < point_cloud_.size(); ++i) {
      // Note that this is a 2D point. The third component is a scaling factor.
      const Eigen::Matrix<T, 3, 1> point((T(point_cloud_[i].position.x())),
                                         (T(point_cloud_[i].position.y())),
                                         T(1.));
      const Eigen::Matrix<T, 3, 1> world = transform * point;
      interpolator.Evaluate(
          (limits.max().x() - world[0]) / limits.resolution() - 0.5 +
              static_cast<double>(kPadding),
          (limits.max().y() - world[1]) / limits.resolution() - 0.5 +
              static_cast<double>(kPadding),
          &residual[i]);
      residual[i] = scaling_factor_ * residual[i];
    }
    return true;
  }
  ```

References

[1]、cartographer代码仓： cartographer-project/cartographer