# SLAM系列之cartographer系统中PoseGraph2D全局位姿优化算法实现分析

这篇文章将向大家介绍2D建图的全局优化算法实现分析。
首先Local SLAM的优化结果会插入到全局优化位姿图中，具体参考如下的代码片段和相关注释分析。

```
//.\cartographer\mapping\internal\global_trajectory_builder.cc
//将Local SLAM的优化结果插入到位姿图中
if (matching_result->insertion_result != nullptr) {
kLocalSlamInsertionResults->Increment();
auto node_id = pose_graph_->AddNode(
    matching_result->insertion_result->constant_data, trajectory_id_,
    matching_result->insertion_result->insertion_submaps);
CHECK_EQ(node_id.trajectory_id, trajectory_id_);
 
//insertion_result的结构体数据定义，constant_data为位姿图节点数据类型
//insertion_submaps为当前局部位姿优化时插入的子地图
//在局部地图更新后返回
struct InsertionResult {
  std::shared_ptr<const TrajectoryNode::Data> constant_data;
  std::vector<std::shared_ptr<const Submap2D>> insertion_submaps;
};
//如下为调用InsertIntoSubmap后返回的数据的具体代码实现  
return absl::make_unique<InsertionResult>(InsertionResult{
  std::make_shared<const TrajectoryNode::Data>(TrajectoryNode::Data{
      time,
      gravity_alignment,
      filtered_gravity_aligned_point_cloud,
      {},  // 'high_resolution_point_cloud' is only used in 3D.
      {},  // 'low_resolution_point_cloud' is only used in 3D.
      {},  // 'rotational_scan_matcher_histogram' is only used in 3D.
      pose_estimate}),
  std::move(insertion_submaps)});  
  
//MatchingResult为Local SLAM优化位姿后的返回结果
struct MatchingResult {
  common::Time time;
  transform::Rigid3d local_pose;
  sensor::RangeData range_data_in_local;
  // 'nullptr' if dropped by the motion filter.
  std::unique_ptr<const InsertionResult> insertion_result;
};  
//如下为LocalTrajectoryBuilder2D::AddRangeData函数的返回结果
//调用代码示意:matching_result = local_trajectory_builder_->AddRangeData
return absl::make_unique<MatchingResult>(
    MatchingResult{time, pose_estimate, std::move(range_data_in_local),
                   std::move(insertion_result)});
```

```
//.\cartographer\mapping\internal\2d\pose_graph_2d.cc
//GetLocalToGlobalTransform函数获取给定trajectory_id的轨迹从其局部坐标系到全局坐标系
//的变换，optimized_pose是新节点在全局坐标系下的位姿函数实现了快速响应和异步处理复杂计算
//的分离，它迅速地将一个新节点的核心信息添加到图中，后将耗时的“计算约束”部分卸载到后台进行。
//AppendNode函数将数据插入到位姿图的数据中
//ComputeConstraintsForNode函数
NodeId PoseGraph2D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  const transform::Rigid3d optimized_pose(
      GetLocalToGlobalTransform(trajectory_id) * constant_data->local_pose);

  const NodeId node_id = AppendNode(constant_data, trajectory_id,
                                    insertion_submaps, optimized_pose);
  // We have to check this here, because it might have changed by the time we
  // execute the lambda.
  const bool newly_finished_submap =
      insertion_submaps.front()->insertion_finished();
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    return ComputeConstraintsForNode(node_id, insertion_submaps,
                                     newly_finished_submap);
  });
  return node_id;
}
//计算轨迹起点相对于全局坐标系的位姿
transform::Rigid3d PoseGraph2D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec2D>& global_submap_poses,
    const int trajectory_id) const {
  //global_submap_poses 包含了所有轨迹的子图，筛选出当前轨迹的
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {//轨迹刚刚开始
    //...刚开始时的位姿初始化方法，如Identity()或其他的估计方法
  }
  //当前轨迹最后一个被优化子图的迭代器。然后通过 ->id 获取该子图的ID
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  //子图的全局位姿与其在轨迹局部坐标系中的自身位姿的逆相乘，来推导出轨迹局部坐标系到全局坐标系的变换
  return transform::Embed3D(
             global_submap_poses.at(last_optimized_submap_id).global_pose) *
         data_.submap_data.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}
```

下面将以问题形式加以说明介绍。
0、全局位姿优化中子图信息表示问题，子图里存储的数据有哪些(子图的位姿为第一给点云帧的优化位姿吗，子图的点云帧序列会和其对应的位姿一起进行存储吗，其位姿保留和第一个点云帧位姿的变换还是直接保存全局位姿？)
1、cartographer中局部SLAM中基于初始预估位姿，点云帧和地图数据基于ceres solver非线性优化问题进行求解，全局SLAM中通过基于ceres solver非线性优化问题进行求解，但基于的数据和算法的具体实现原理有哪些不同？
2、回环检测算法如何实现，作为约束放入到位姿图中，同时也相对于做了回环校正对吗？
3、位姿图优化 (Pose Graph Optimization, PGO) 的算法原理详细介绍

如果几乎每一个扫描帧的位姿都插入到位姿图节点中，会不会位姿图过大？

ComputeConstraintsForNode函数是逻辑分析的关键。

```
//
struct PoseGraphData {
  // Submaps get assigned an ID and state as soon as they are seen, even
  // before they take part in the background computations.如果
  // const SubmapId submap_id =
  //   data_.submap_data.Append(trajectory_id, InternalSubmapData());
  //   data_.submap_data.at(submap_id).submap = insertion_submaps.back();
  MapById<SubmapId, InternalSubmapData> submap_data;

  // Global submap poses currently used for displaying data.
  MapById<SubmapId, optimization::SubmapSpec2D> global_submap_poses_2d;
  MapById<SubmapId, optimization::SubmapSpec3D> global_submap_poses_3d;

  // Data that are currently being shown.
  //  const NodeId node_id = data_.trajectory_nodes.Append(
  //    trajectory_id, TrajectoryNode{constant_data, optimized_pose});
  MapById<NodeId, TrajectoryNode> trajectory_nodes;

  // Global landmark poses with all observations.
  std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
      landmark_nodes;

  // How our various trajectories are related.
  TrajectoryConnectivityState trajectory_connectivity_state;
  int num_trajectory_nodes = 0;
  std::map<int, InternalTrajectoryState> trajectories_state;

  // Set of all initial trajectory poses.
  std::map<int, PoseGraph::InitialTrajectoryPose> initial_trajectory_poses;

  std::vector<PoseGraphInterface::Constraint> constraints;
};
```

```
//.\cartographer\mapping\internal\optimization\optimization_problem_2d.cc
//node_data_为存储了位姿图所有节点信息的容器数据，位姿图中的节点为点云帧对应的位姿，AddParameterBlock函数
//定义了优化的变量，AddParameterBlock()是Ceres的核心方法，用于将变量(这里是机器人的2D位姿,维度是3即x,y,yaw)
//注册为优化参数。如果冻结该组优化变量，则需要调用SetParameterBlockConstant将其设置为设置其为const
//这里子图的全局位姿和点云帧节点的全局位姿将作为优化变量同时进行优化，后面有相关约束的定义
for (const auto& submap_id_data : submap_data_) {
  const bool frozen =
      frozen_trajectories.count(submap_id_data.id.trajectory_id) != 0;
  C_submaps.Insert(submap_id_data.id,
                    FromPose(submap_id_data.data.global_pose));
  problem.AddParameterBlock(C_submaps.at(submap_id_data.id).data(), 3);
  if (first_submap || frozen) {
    first_submap = false;
    // Fix the pose of the first submap or all submaps of a frozen
    // trajectory.
    problem.SetParameterBlockConstant(C_submaps.at(submap_id_data.id).data());
  }
}
for (const auto& node_id_data : node_data_) {
  const bool frozen =
      frozen_trajectories.count(node_id_data.id.trajectory_id) != 0;
  C_nodes.Insert(node_id_data.id, FromPose(node_id_data.data.global_pose_2d));
  problem.AddParameterBlock(C_nodes.at(node_id_data.id).data(), 3);
  if (frozen) {
    problem.SetParameterBlockConstant(C_nodes.at(node_id_data.id).data());
  }
}
```


```
//.cartographer\mapping\internal\optimization\optimization_problem_2d.cc
//AddResidualBlock是ceres中的核心方法，用于将一个误差项（即残差块）添加到优化问题中
//通过向问题中添加残差块的方式添加优化问题求解的约束
//INTER_SUBMAP表示这是子图间的回环闭合约束，INTRA_SUBMAP为子图内里程计约束等
//SpaCostFunction是稀疏位姿调整(Sparse Pose Adjustment,SPA)问题的CostFunction
//在2D位姿图优化中，SpaCostFunction通常计算两个2D位姿之间的相对变换与观测值constraint.pose
//之间的误差测量值constraint.pose来自SLAM前端，如IMU预积分，或里程计得出的位姿变化增量，
//这里为基于局部SLAM的子图内点云帧和子图位姿(初始第一个点云帧)的相对位姿变化量作为约束值
//C_submaps.at(constraint.submap_id).data()返回的是子图全局位姿，也是起始位姿
//C_nodes.at(constraint.node_id).data());返回的是节点位姿，亦即当前点云帧位姿，也为end位姿
//约束的过程即基于测量值来调整全局位姿，使得整个位姿图的内部一致性达到最佳。
for (const Constraint& constraint : constraints) {
  problem.AddResidualBlock(
      CreateAutoDiffSpaCostFunction(constraint.pose),
      // Loop closure constraints should have a loss function.
      constraint.tag == Constraint::INTER_SUBMAP
          ? new ceres::HuberLoss(options_.huber_scale())
          : nullptr,
      C_submaps.at(constraint.submap_id).data(),
      C_nodes.at(constraint.node_id).data());
}
```

```
//lambda函数在后台线程计算节点的约束,并根据配置选项的条件是否满足来
//运行基于位姿图的全局优化算法。
AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    return ComputeConstraintsForNode(node_id, insertion_submaps,
                                     newly_finished_submap);
//    
void PoseGraph2D::AddWorkItem(
    const std::function<WorkItem::Result()>& work_item) {
  absl::MutexLock locker(&work_queue_mutex_);
  if (work_queue_ == nullptr) {
    //......
  }
  work_queue_->push_back({now, work_item});
}
```

References

[1]、cartographer代码仓： cartographer-project/cartographer