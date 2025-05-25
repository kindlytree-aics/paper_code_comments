# range_data_inserter_3d

range_data_inserter_3d 组件负责将原始的 3D 传感器扫描数据整合到 Cartographer 的 3D 栅格地图表示中。
主要功能是将 3D 传感器（如激光雷达）的测量数据（RangeData）插入到 3D 栅格地图（HybridGrid）中,并可选地更新强度信息到强度栅格地图（IntensityHybridGrid)

```c++
namespace cartographer {
namespace mapping {

// 从 Lua 参数字典创建 RangeDataInserter3D 的配置选项。
proto::RangeDataInserterOptions3D CreateRangeDataInserterOptions3D(
    common::LuaParameterDictionary* parameter_dictionary);

// RangeDataInserter3D 类负责将传感器范围数据（RangeData）插入到 3D 混合栅格地图（HybridGrid）中。
class RangeDataInserter3D {
 public:
  // 构造函数，接收配置选项。
  explicit RangeDataInserter3D(
      const proto::RangeDataInserterOptions3D& options);

  // 删除拷贝构造函数，防止对象被复制。
  RangeDataInserter3D(const RangeDataInserter3D&) = delete;
  // 删除拷贝赋值运算符，防止对象被赋值。
  RangeDataInserter3D& operator=(const RangeDataInserter3D&) = delete;

  // 将 'range_data' 插入到 'hybrid_grid' 中，并可选地插入到 'intensity_hybrid_grid' 中。
  // hybrid_grid: 指向存储占用概率的混合栅格地图的指针。
  // intensity_hybrid_grid: 指向存储强度信息的混合栅格地图的指针（可选，可为 nullptr）。
  void Insert(const sensor::RangeData& range_data, HybridGrid* hybrid_grid,
              IntensityHybridGrid* intensity_hybrid_grid) const;

 private:
  // 存储 RangeDataInserter3D 的配置选项。
  const proto::RangeDataInserterOptions3D options_;
  // 预先计算的查找表，用于快速更新栅格单元被击中（hit）时的概率。
  const std::vector<uint16> miss_table_;
  // 预先计算的查找表，用于快速更新栅格单元未被击中（miss）时的概率。
  const std::vector<uint16> miss_table_;
};

}  // namespace mapping
}  // namespace cartographer

```


## 关于miss_table_以及miss_table_

hit_table_ 和 miss_table_ 是根据配置中的命中概率 ( hit_probability ) 或未命中概率 ( miss_probability ) 计算得到的。它们存储的是将配置的概率（转换为 Odds 形式）应用到栅格单元当前概率值上之后，得到的新概率值（以特定的整数形式编码）。
```c++
RangeDataInserter3D::RangeDataInserter3D(
    const proto::RangeDataInserterOptions3D& options)
    : options_(options),
      hit_table_(
          ComputeLookupTableToApplyOdds(Odds(options_.hit_probability()))),
      miss_table_(
          ComputeLookupTableToApplyOdds(Odds(options_.miss_probability()))){}
```
### table_是如何进行创建的

```c++
  // Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds()
  // to the probability of the cell at 'index' if the cell has not already been
  // updated. Multiple updates of the same cell will be ignored until
  // FinishUpdate() is called. Returns true if the cell was updated.
  //
  // If this is the first call to ApplyOdds() for the specified cell, its value
  // will be set to probability corresponding to 'odds'.
/**
 * @brief 计算用于应用给定赔率 (odds) 的查找表。
 *
 * 该函数生成一个查找表，用于高效地将概率更新操作应用到栅格地图的每个单元格。
 * 查找表的索引为当前单元格的 uint16 值（0~32767），
 * 表中的每个元素是应用了输入赔率 odds 后的新概率值（uint16），并加上了更新标记 kUpdateMarker。
 *
 * 具体流程如下：
 * 1. 对于未知单元格（索引0），直接将 odds 转换为概率值并编码为 uint16，最后加上 kUpdateMarker。
 * 2. 对于已知单元格（索引1~32767），先查表获取当前概率值，然后将 odds 与当前概率对应的赔率相乘，得到新的赔率，
 *    再将其转换为概率值并编码为 uint16，最后加上 kUpdateMarker。
 *
 * 这样可以避免在每次更新时重复进行复杂的概率与赔率转换运算，提高效率。
 *
 * @param odds 需要应用的赔率（如 hit_probability 或 miss_probability 转换而来）。
 * @return std::vector<uint16> 查找表，长度为 kValueCount，每个元素为更新后的概率值（带更新标记）。
 */
std::vector<uint16> ComputeLookupTableToApplyOdds(const float odds) {
  std::vector<uint16> result;
  // 预分配空间，提升效率。kValueCount 通常为 32768。
  result.reserve(kValueCount);
  // 对于未知单元格（索引0），直接用 odds 计算概率并编码。
  result.push_back(ProbabilityToValue(ProbabilityFromOdds(odds)) +
                   kUpdateMarker);
  // 对于已知单元格（索引1~kValueCount-1），查表获取当前概率，计算新赔率并更新。
  for (int cell = 1; cell != kValueCount; ++cell) {
    // 当前单元格的概率值
    float current_probability = (*kValueToProbability)[cell];
    // 当前概率对应的赔率
    float current_odds = Odds(current_probability);
    // 新赔率 = 输入赔率 * 当前赔率
    float new_odds = odds * current_odds;
    // 新概率 = 概率函数（新赔率）
    float new_probability = ProbabilityFromOdds(new_odds);
    // 编码为 uint16 并加上更新标记
    result.push_back(ProbabilityToValue(new_probability) + kUpdateMarker);
  }
  return result;
}
```

### 更新table

```c++
  // Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds()
  // to the probability of the cell at 'index' if the cell has not already been
  // updated. Multiple updates of the same cell will be ignored until
  // FinishUpdate() is called. Returns true if the cell was updated.
  //
  // If this is the first call to ApplyOdds() for the specified cell, its value
  // will be set to probability corresponding to 'odds'.
  // 应用在调用 ComputeLookupTableToApplyOdds() 时指定的 'odds'（赔率）
  // 到 'index' 处单元格的概率上，前提是该单元格尚未被更新。
  // 对同一单元格的多次更新将被忽略，直到调用 FinishUpdate()。
  // 如果单元格被更新，则返回 true。
  //
  // 如果这是对指定单元格的第一次调用 ApplyLookupTable()，其值将被设置为
  // 对应于 'odds' 的概率。
  // index: 要更新的单元格的三维索引。
  // table: 预先计算好的查找表（hit_table 或 miss_table）。
  bool ApplyLookupTable(const Eigen::Array3i& index,
                        const std::vector<uint16>& table) {
    // 检查查找表的大小是否等于更新标记值（这是一个内部一致性检查）。
    DCHECK_EQ(table.size(), kUpdateMarker);
    // 获取指向指定索引处单元格值的可变指针。
    uint16* const cell = mutable_value(index);
    // 检查该单元格是否已经在本轮更新中被标记过（值大于等于 kUpdateMarker）。
    if (*cell >= kUpdateMarker) {
      // 如果已经更新过，则直接返回 false，不做任何操作。
      return false;
    }
    // 将该单元格的指针添加到 update_indices_ 列表中，以便后续 FinishUpdate() 处理。
    update_indices_.push_back(cell);
    // 使用查找表更新单元格的值。table[*cell] 会根据当前值 *cell 查找到新的值。
    // 新值包含了更新后的概率以及更新标记。
    *cell = table[*cell];
    // 检查更新后的值是否确实包含了更新标记。
    DCHECK_GE(*cell, kUpdateMarker);
    // 返回 true，表示单元格已被成功更新。
    return true;
  }
```


## RangeData插入的关键细节

```c++
void RangeDataInserter3D::Insert(
    const sensor::RangeData& range_data, HybridGrid* hybrid_grid,
    IntensityHybridGrid* intensity_hybrid_grid) const {
  CHECK_NOTNULL(hybrid_grid);

  // 遍历所有激光点的返回（即击中障碍物的点）
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    // 计算该点在体素网格中的索引
    const Eigen::Array3i hit_cell = hybrid_grid->GetCellIndex(hit.position);
    // 使用命中查找表（hit_table_）更新该体素的概率值
    // 本质上是将该体素的概率提升（更倾向于有障碍物）
    hybrid_grid->ApplyLookupTable(hit_cell, hit_table_);
  }
  
  // 不在插入miss（空闲）前开启新一轮更新，保证hit优先级更高
  // （即如果同一个体素既有hit又有miss，只保留hit的更新）
  InsertMissesIntoGrid(miss_table_, range_data.origin, range_data.returns,
                       hybrid_grid, options_.num_free_space_voxels());

  // 如果有强度网格，则插入强度信息
  if (intensity_hybrid_grid != nullptr) {
    InsertIntensitiesIntoGrid(range_data.returns, intensity_hybrid_grid,
                              options_.intensity_threshold());
  }
  // 本轮所有体素更新完成，清除更新标记
  hybrid_grid->FinishUpdate();
}
```

```c++
 hybrid_grid->ApplyLookupTable(hit_cell, hit_table_);
```
这里对于每一个激光点命中的体素（hit_cell），通过查找表 hit_table_ 来更新该体素的概率值。这个查找表实际上是根据命中概率（hit probability）预先计算好的，能够将体素的原始概率值映射为更新后的概率值，从而反映出该体素被观测为“有障碍物”的概率增加。



其中调用了`InsertMissesIntoGrid`和`InsertIntensitiesIntoGrid`这两个工厂辅助函数进行插入miss以及强度信息


- InsertMissesIntoGrid
```c++
// 将传感器射线未击中的信息（misses）插入到混合栅格地图中。
// miss_table: 用于更新 miss 概率的查找表。
// origin: 传感器原点坐标。
// returns: 传感器返回的点云数据（包含击中点）。
// hybrid_grid: 需要更新的混合栅格地图。
// num_free_space_voxels: 为了性能，只更新靠近击中点一定数量的空闲空间体素。
void InsertMissesIntoGrid(const std::vector<uint16>& miss_table,
                          const Eigen::Vector3f& origin,
                          const sensor::PointCloud& returns,
                          HybridGrid* hybrid_grid,
                          const int num_free_space_voxels) {
  // 获取传感器原点在栅格地图中的索引。
  const Eigen::Array3i origin_cell = hybrid_grid->GetCellIndex(origin);
  // 遍历所有击中点。
  for (const sensor::RangefinderPoint& hit : returns) {
    // 获取击中点在栅格地图中的索引。
    const Eigen::Array3i hit_cell = hybrid_grid->GetCellIndex(hit.position);

    // 计算从原点到击中点的栅格索引差值。
    const Eigen::Array3i delta = hit_cell - origin_cell;
    // 计算在变化最快的维度上，从原点到击中点需要采样的数量。
    // 这样可以确保在两个样本之间，至少跨越一个体素。
    const int num_samples = delta.cwiseAbs().maxCoeff();
    // 检查样本数量是否在合理范围内。
    CHECK_LT(num_samples, 1 << 15);
    // 'num_samples' 是我们在 'origin' 和 'hit' 之间的直线上等距放置的样本数
    // （包括用于子体素的小数部分）。它的选择是为了在两个样本之间，
    // 我们在变化最快的维度上从一个体素移动到下一个体素。
    //
    // 为了性能，只更新最后的 'num_free_space_voxels' 个体素。
    // 遍历从原点到击中点射线路径上的体素（靠近击中点的部分）。
    for (int position = std::max(0, num_samples - num_free_space_voxels);
         position < num_samples; ++position) {
      // 计算当前 miss 体素的索引。
      const Eigen::Array3i miss_cell =
          origin_cell + delta * position / num_samples;
      // 使用 miss 查找表更新该体素的概率，表示该空间是空闲的。
      hybrid_grid->ApplyLookupTable(miss_cell, miss_table);
    }
  }
}

```

- InsertIntensitiesIntoGrid

```c++

// 将点云的强度信息插入到强度混合栅格地图中。
// returns: 传感器返回的点云数据。
// intensity_hybrid_grid: 需要更新的强度混合栅格地图。
// intensity_threshold: 强度阈值，低于此阈值的强度才会被记录。
void InsertIntensitiesIntoGrid(const sensor::PointCloud& returns,
                               IntensityHybridGrid* intensity_hybrid_grid,
                               const float intensity_threshold) {
  // 检查点云是否包含强度信息。
  if (returns.intensities().size() > 0) {
    // 遍历所有点。
    for (size_t i = 0; i < returns.size(); ++i) {
      // 如果强度值高于阈值，则跳过。
      if (returns.intensities()[i] > intensity_threshold) {
        continue;
      }
      // 获取击中点在强度栅格地图中的索引。
      const Eigen::Array3i hit_cell =
          intensity_hybrid_grid->GetCellIndex(returns[i].position);
      // 将强度值添加到对应的强度栅格单元。
      intensity_hybrid_grid->AddIntensity(hit_cell, returns.intensities()[i]);
    }
  }
}
```