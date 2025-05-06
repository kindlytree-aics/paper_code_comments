# mapping/3d的数据结构表示以及算法实现


## hybrid_grid.h

### FlatGrid (平面网格/一维)

- FlatGrid是Cartographer混合网格系统的基础组件，为局部高分辨率环境表示提供了高效的内存存储方案它通常与NestedGrid和DynamicGrid配合使用，形成完整的HybridGrid数据结构。

这是一个模板类，接受两个参数：
- TValueType ：网格单元存储的数据类型
- kBits ：决定网格大小的位参数（网格大小为2^kBits × 2^kBits × 2^kBits）

```c++
// A flat grid of '2^kBits' x '2^kBits' x '2^kBits' voxels storing values of
// type 'ValueType' in contiguous memory. Indices in each dimension are 0-based.
template <typename TValueType, int kBits>
class FlatGrid {
...
```

数据存放

```c++
  std::array<ValueType, 1 << (3 * kBits)> cells_;
```
- 使用std::array存储所有网格单元数据
- 总单元数为2^(3*kBits)，也就是每个维度有2^kBits个单元


关键方法

- value() ：获取指定索引处的值
- mutable_value() ：获取可修改的指定索引处值的指针
- 使用 ToFlatIndex() 将3D索引转换为线性索引（z-major顺序）

```c++
// Converts an 'index' with each dimension from 0 to 2^'bits' - 1 to a flat
// z-major index.
inline int ToFlatIndex(const Eigen::Array3i& index, const int bits) {
  DCHECK((index >= 0).all() && (index < (1 << bits)).all()) << index;
  return (((index.z() << bits) + index.y()) << bits) + index.x();
}
```

### NestedGrid(二维)
NestedGrid（嵌套网格）数据结构，是3D SLAM系统中混合网格的重要组成部分,由多个FlatGrid组成的层级结构

这也是一个模板类，接受两个参数：
- WrappedGrid ：被包装的网格类型（通常是FlatGrid）
- kBits ：决定网格层级的位参数

```c++
// A grid consisting of '2^kBits' x '2^kBits' x '2^kBits' grids of type
// 'WrappedGrid'. Wrapped grids are constructed on first access via
// 'mutable_value()'.
template <typename WrappedGrid, int kBits>
```

数据存放:
```c++
std::array<std::unique_ptr<WrappedGrid>, 1 << (3 * kBits)> meta_cells_;
```

关键方法 ：
- value() ：读取指定索引处的值
- mutable_value() ：获取可修改的指定索引处值的指针
- 使用 GetMetaIndex() 计算子网格索引


### DynamicGrid (动态网格/三维)

模板类，接受一个参数 WrappedGrid （通常为NestedGrid),是由NestedGrid组成的可扩展结构

- 使用分层结构管理网格单元，初始为2x2x2网格
- 支持动态扩展以适应更大范围的环境

核心设计特点 ：
- 惰性初始化策略（首次访问时创建子网格）
- 支持负索引，网格中心位于(0,0,0)
- 使用 bits_ 记录当前网格层级
- 通过 meta_cells_ 存储子网格指针
 
 关键方法 ：
- value() ：读取指定索引处的值
- mutable_value() ：获取可修改的指定索引处值的指针
- Grow() ：将网格尺寸扩大一倍

```c++
  // Grows this grid by a factor of 2 in each of the 3 dimensions.
  void Grow() {
    const int new_bits = bits_ + 1;
    CHECK_LE(new_bits, 8);
    std::vector<std::unique_ptr<WrappedGrid>> new_meta_cells_(
        8 * meta_cells_.size());
    for (int z = 0; z != (1 << bits_); ++z) {
      for (int y = 0; y != (1 << bits_); ++y) {
        for (int x = 0; x != (1 << bits_); ++x) {
          const Eigen::Array3i original_meta_index(x, y, z);
          const Eigen::Array3i new_meta_index =
              original_meta_index + (1 << (bits_ - 1));
          new_meta_cells_[ToFlatIndex(new_meta_index, new_bits)] =
              std::move(meta_cells_[ToFlatIndex(original_meta_index, bits_)]);
        }
      }
    }
    meta_cells_ = std::move(new_meta_cells_);
    bits_ = new_bits;
  }
```
grow是一个将3D网格在每个维度上扩展2倍的方法

- 将当前网格分辨率从2^bits提升到2^(bits+1)
- 重新组织子网格数据到新的内存布局
- 保持原有数据的正确位置关系


# 总结

也就是说这三种网格实现是如下实现的

- FlatGrid → NestedGrid(包含多个FlatGrid) → DynamicGrid(包含多个NestedGrid)


这样的设计优势：

- 内存效率（稀疏区域不分配）
- 访问效率（局部性原理）
- 动态扩展能力（适应SLAM需求）
- 支持从高分辨率局部到低分辨率全局的统一表示

典型应用：
- FlatGrid：局部高精度地图
- NestedGrid：中等规模环境
- DynamicGrid：大规模SLAM建图
