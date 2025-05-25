# TF介绍
TF(TransForm)，就是坐标转换，包括了位置和姿态两个方面的变换,而ROS的TF系统是一个用于跟踪多个坐标系之间关系的库

## TF原理


### ROS TF主要实现机制

主要实现机制包括：

(1) 数据结构：

- 使用树状结构存储坐标系关系
- 每个变换包含时间戳，支持时间查询
- 使用四元数表示旋转，向量表示平移
(2) 核心组件：

- TransformBroadcaster：发布坐标变换
- TransformListener：订阅并缓存坐标变换
- BufferCore：存储和管理变换数据
(3) 工作机制：

- 分布式：各节点可以独立发布/订阅变换
- 缓存机制：保存历史变换数据，支持时间查询
- 自动维护：自动处理坐标系树的关系和更新
(4) 消息流程：

- 发布者通过 /tf 或 /tf_static 话题发布变换
- 监听者订阅这些话题并更新本地缓存
- 应用通过Buffer API查询特定时间的变换
(5) 时间处理：

- 支持时间同步和插值
- 可以查询过去或预测未来的变换
- 处理变换的时间有效性

### TF树的建立
tf是一个树状结构，维护坐标系之间的关系，靠话题通信机制来持续地发布不同link之间的坐标关系。
在开始建立TF树的时候需要指定第一个父坐标系（parent frame）作为最初的坐标系。比如机器人系统中的map坐标系。

在第一次发布一个从已有的parent frame到新的child frame的坐标系变换时，这棵树就会添加一个树枝，之后就是维护。

TF树的建立和维护靠的是tf提供的tfbroadcastor类的sendtransform接口。

transformBroadcaster()类就是一个publisher,而sendTransform的作用是来封装publish的函数

## 总结
TF整体是一个树,其中,tfbroadcastor的类里有个publisher而tflisener的类里有个subscriber，一个发布叫/tf的topic，一个订阅这个topic，传送的消息message里包含了每一对parent frameid和child frameid的信息。这个机制意味着，所有的tb会发布某一特定的parent到child的变换，而所有tl会收到所有的这些变换，然后tl利用一个tfbuffercore的数据结构维护一个完整的树结构及其状态。基于此，tl在使用这棵树时，会用lookuptransform或waitfortransform来获得任意坐标系之间的变换。

