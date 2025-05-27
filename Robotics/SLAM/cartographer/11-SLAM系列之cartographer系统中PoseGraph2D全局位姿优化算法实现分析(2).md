# SLAM系列之cartographer系统中PoseGraph2D全局位姿优化算法实现分析

这篇文章将向大家介绍2D建图的全局优化算法实现分析。
首先Local SLAM的优化结果会插入到全局优化位姿图中，具体参考如下的代码片段和相关注释分析。


篇文章主要向大家分析介绍了cartographer系统中基于ceres的位姿图构建的相关方面的细节，本篇将继续cartographer系统中关于全局优化的流程处理中的更多细节问题。也将继续以问答的形式向大家做分析介绍。

问题1：cartographer的实现中，一般一个点云帧会对应到两个子图中，在动态构建位姿图的过程中，是否是两个子图的全局位姿节点和点云帧位姿节点都要建立约束呢？

回答：计算约束的函数为PoseGraph2D::ComputeConstraintsForNode，在该函数中根据三种情况来计算约束：(1)、根据当前点云帧节点(简称节点)和其相关的正在构建的两个局部地图的全局位姿参数块节点(简称子图节点)之间的约束(INTRA_SUBMAP)；(2)、当前点云帧节点和已经构建完成的子图(标记为完成状态)之间可能的回环约束;（3）、如果当前的子图也刚刚标记为结束状态，则需要更新已有的位姿图点云帧位姿节点和当前子图的约束，其中后面两种约束为INTER_MAP回环约束，将在下一个问题中加以具体的描述。

问题2：回环检测和约束如何实现？

回环检测的算法在PoseGraph2D::ComputeConstraint函数中进行的实现，其计算涉及到了轨迹内部和轨迹之间的可能的回环检测。具体会根据条件选择局部匹配搜索和全局匹配搜索。关于其约束构建的具体计算参考ComputeConstraint，下面对其具体的细节做相关分析说明。

```
//.\cartographer\mapping\internal\constraints\constraint_builder_2d.cc
//ComputeConstraint(node_id, submap_id); Compute'pose_estimate'in three stages:
//1.Fast estimate using the fast correlative scan matcher.
//2.Prune if the score is too low. 3. Refine.
//在其调用的地方会遍历所有可能的子图(finished_submap_ids)依此进行搜索计算。
//快速初步估计(Fast Correlative Scan Matching)
//在基于单个子图内在较大的搜索空间或粗略的范围内找到一个较好的初始匹配。
if (match_full_submap) {
kGlobalConstraintsSearchedMetric->Increment();
  //MatchFullSubmap会在整个子图的范围内进行搜索，而不是局限于一个小范围
  if (submap_scan_matcher.fast_correlative_scan_matcher->MatchFullSubmap(
          constant_data->filtered_gravity_aligned_point_cloud,
          options_.global_localization_min_score(), &score, &pose_estimate)) {
	//......
  } else {
    return;
  }
} else {
  kConstraintsSearchedMetric->Increment();
  //initial_pose作为起始点。只会在initial_pose周围的一个较小范围内进行搜索，而不是整个子图
  if (submap_scan_matcher.fast_correlative_scan_matcher->Match(
          initial_pose, constant_data->filtered_gravity_aligned_point_cloud,
          options_.min_score(), &score, &pose_estimate)) {
    //......
  } else {
    return;
  }
}
{
  absl::MutexLock locker(&mutex_);
  score_histogram_.Add(score);
}

//Use the CSM estimate as both the initial and previous pose. This has the effect that,
//in the absence of better information, we prefer the original CSM estimate.
//基于CSM(Correlative Scan Matching)初步估计作为起点，通过非线性优化方法进行高精度的姿态精调。
ceres::Solver::Summary unused_summary;
ceres_scan_matcher_.Match(pose_estimate.translation(), pose_estimate,
                        constant_data->filtered_gravity_aligned_point_cloud,
                        *submap_scan_matcher.grid, &pose_estimate,
                        &unused_summary);
```

其中计算这些constraints的过程也是作为异步背景线程中来运行的。具体更多的实现可以参考ConstraintBuilder2D类中的更多的代码实现细节。

问题3：全局地图的表示以及更新的方法细节如何实现？位姿图进行全局优化后如何更新地图的表示，以及如何更新子地图和点云帧的全局位姿信息？

回答：cartographer_node是cartographer在ROS中最主要的运行节点，在其内部运行了cartographer核心库的实现，包括SLAM的前端和后端，核心库本身不维护全局唯一的大图，但是其会已优化好的子图数据（包括每个子图的网格内容及其最终优化后的全局姿态）以二进制流（pbstream 格式）的形式发送给接收融合节点，如cartographer_occupancy_grid_node，在接收融合节点内会创建一个足够大的消息nav_msgs/OccupancyGrid，并将子地图进行融合后，以/map的ros topic发布出来。

问题4：位姿图优化一般作为背景线程，在什么时候被触发？如何支持多轨迹联合全局优化？

回答：在cartographer的内部实现中，基于线程池和任务队列的方式进行位姿图节点的更新，约束的计算等计算任务较重的计算逻辑，在cartographer中节点(点云帧和子图都有对应的trajectory_id属性，在进行全局优化的时候，不同轨迹的点云帧节点和子图数据更新插入到位姿图的逻辑是统一的接口。位姿图优化的是所有轨迹的节点和子子图节点和约束边的整体全局多轨迹位姿图。在全局的多轨迹地图点云节点联合位姿优化中，已经冻结的轨迹中的点云帧节点和子地图节点在位姿图节点中将会被设置为常量，不参与进一步的优化，也有利于效率的进一步提高。

References
[1]、cartographer代码仓： cartographer-project/cartographer