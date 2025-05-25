# collator源码分析

基于`collator__interface`基类实现,基类提供以下函数来进行重载

传感器数据进入 Cartographer 核心处理流程的统一入口或协调器。

Collator （字面含义为整理者），或类似组件主要负责解决底层传感器数据的异步和乱序问题，通过缓冲和排序，产生一个统一的、时间严格递增的传感器数据流。它是原始传感器数据的第一个汇集点和排序器

TrajectoryCollator (或类似组件) 接收这个时间有序的数据流，并根据轨迹 ID 将数据分发给相应的 TrajectoryBuilder。它是将数据送入具体 SLAM 算法处理单元的统一入口和调度器。

```c++
virtual void AddTrajectory(
      int trajectory_id,
      const absl::flat_hash_set<std::string>& expected_sensor_ids,
      const Callback& callback) = 0;

  // Marks 'trajectory_id' as finished.
  virtual void FinishTrajectory(int trajectory_id) = 0;

  // Adds 'data' for 'trajectory_id' to be collated. 'data' must contain valid
  // sensor data. Sensor packets with matching 'data.sensor_id_' must be added
  // in time order.
  virtual void AddSensorData(int trajectory_id, std::unique_ptr<Data> data) = 0;

  // Dispatches all queued sensor packets. May only be called once.
  // AddSensorData may not be called after Flush.
  virtual void Flush() = 0;

  // Must only be called if at least one unfinished trajectory exists. Returns
  // the ID of the trajectory that needs more data before CollatorInterface is
  // unblocked. Returns 'nullopt' for implementations that do not wait for a
  // particular trajectory.
  virtual absl::optional<int> GetBlockingTrajectoryId() const = 0;
 ```

## 多轨迹传感器队列管理（AddTrajectory）
- 作用：
为指定轨迹初始化多个传感器队列，每个传感器ID对应一个独立队列，绑定callback回调函数处理数据。

- 系统支持：
支持多机器人协同建图（不同轨迹）和多传感器配置，隔离不同来源数据，避免交叉污染。

## 轨迹结束处理（FinishTrajectory）
- 作用：
标记某轨迹的所有传感器队列为完成状态，触发队列清理。

- 系统支持：
在轨迹建图完成或任务终止时释放资源，防止内存泄漏，维护系统资源效率。

## 传感器数据路由（AddSensorData）
- 作用：
根据数据中的轨迹ID和传感器ID，将数据路由到对应的队列。

- 系统支持：
实现数据自动分类，确保激光雷达、IMU等异构数据进入正确处理管道，支撑多传感器融合。

## 强制数据处理（Flush）
- 作用：
立即处理所有队列中的剩余数据，清空缓存。

- 系统支持：
在系统关闭、重置或需要同步状态时确保数据完整性，避免残留未处理数据导致状态不一致。

## 阻塞检测（GetBlockingTrajectoryId）
- 作用：
返回因数据不足导致处理阻塞的轨迹ID。

- 系统支持：
辅助监控系统状态，快速定位数据缺失的轨迹，便于调试或触发容错机制。
