# ROS2 humble下的Cartographer编译指南

- 系统配置
1. WSL2下的Ubuntu22.04
2. ROS2 humble版本



### 安装ROS2 humble

```bash
wget http://fishros.com/install -O fishros && . fishros
```

- 选择===> 一键安装:ROS(支持ROS和ROS2,树莓派Jetson)
- 选择===> 更换系统源再继续安装
- 选择===> humble系统
- 选择===> humble系统桌面版

### ROS2的rosdepc安装

```bash
wget http://fishros.com/install -O fishros && . fishros
```

选择3,一键配置rosdepc。安装完成后运行`rosdepc update`




## ROS2的cartographer源码安装编译

- 创建工作空间
```bash
mkdir -p carto_ws/src && cd carto_ws/src
```

- 克隆代码

```bash
git clone https://github.com/ros2/cartographer.git -b ros2
git clone https://github.com/ros2/cartographer_ros.git -b ros2
```

- 安装依赖

```bash
cd ~/carto_ws
rosdepc install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
```

- 编译

```
colcon build --packages-up-to cartographer_ros
```

- 验证成功

```bash
source install/setup.bash
ros2 pkg list | grep cartographer
```

- 如下则成功

```bash
cartographer_ros
cartographer_ros_msgs
```