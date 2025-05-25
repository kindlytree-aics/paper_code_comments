# Cartographer以及Cartographer ROS编译指南

- 在安装Cartographer ROS前,需安装Cartographer

## Cartographer安装依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    clang \
    cmake \
    g++ \
    git \
    google-mock \
    libboost-all-dev \
    libcairo2-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblua5.2-dev \
    libsuitesparse-dev \
    lsb-release \
    ninja-build \
    python3-sphinx \
    stow

# Install Protocol Buffers and Abseil if available.
# No need to build it ourselves.
case "$(lsb_release -sc)" in
    jammy|bullseye)
        sudo apt-get install -y libgmock-dev protobuf-compiler libabsl-dev ;;
    focal|buster)
        sudo apt-get install -y libgmock-dev protobuf-compiler ;;
    bionic)
        ;;
esac
```

## 安装abseil-cpp

```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
git checkout 215105818dfde3174fe799600bb0f3cae233d0bf # 20211102.0
mkdir build
cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/stow/absl \
  ..
ninja
sudo ninja install
cd /usr/local/stow
sudo stow absl
```

- 如若在`git checkout`时有报错,按照提示输入即可

```bash
VERSION="v3.4.1"

# Build and install proto3.
git clone https://github.com/google/protobuf.git
cd protobuf
git checkout tags/${VERSION}
mkdir build
cd build
cmake -G Ninja \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -Dprotobuf_BUILD_TESTS=OFF \
  ../cmake
ninja
sudo ninja install
```

- 编译cartographer

```bash
# Build and install Cartographer.
git clone https://github.com/cartographer-project/cartographer.git
cd cartographer
mkdir build
cd build
cmake .. -G Ninja
ninja
CTEST_OUTPUT_ON_FAILURE=1 ninja test
sudo ninja install
```


## Cartographer ROS 编译

- 支持ROS版本为 `Melodic`/`Noetic`

### 安装依赖


在带有 ROS Noetic 的 Ubuntu Focal 上，使用以下命令安装上述工具：

```bash
sudo apt-get update
sudo apt-get install -y python3-wstool python3-rosdep ninja-build stow
```
在较旧的发行版中：

```bash
sudo apt-get update
sudo apt-get install -y python-wstool python-rosdep ninja-build stow
```
安装工具后，在“catkin_ws”中创建一个新的cartographer_ros工作区。
```bash
mkdir catkin_ws
cd catkin_ws
wstool init src
wstool merge -t src https://raw.githubusercontent.com/cartographer-project/cartographer_ros/master/cartographer_ros.rosinstall
wstool update -t src
```

现在您需要安装 cartographer_ros 的依赖项。首先，我们使用rosdep安装所需的软件包。如果您在安装 ROS 后已经执行过命令“sudo rosdep init”，则该命令将打印错误。此错误可以忽略。

```bash
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y
```

Cartographer 使用abseil-cpp库，需要使用此脚本手动安装：

```bash
src/cartographer/scripts/install_abseil.sh
```
由于版本冲突，你可能需要使用以下方法卸载 ROS abseil-cpp

```bash
sudo apt-get remove ros-${ROS_DISTRO}-abseil-cpp
```

构建并安装。

```bash
catkin_make_isolated --install --use-ninja

source ~/catkin_ws/install_isolated/setup.bash
```

- 也可以加入至 .bashrc中

echo source ~/catkin_ws/install_isolated/setup.bash >> ~/.bashrc


### 运行示例


- 下载2D数据包并播放

```bash
wget -P ~/Downloads https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/cartographer_paper_deutsches_museum.bag

roslaunch cartographer_ros demo_backpack_2d.launch bag_filename:=${HOME}/Downloads/cartographer_paper_deutsches_museum.bag
```

- 下载3D数据包并播放(该数据包较大)

```bash
wget -P ~/Downloads https://storage.googleapis.com/cartographer-public-data/bags/backpack_3d/with_intensities/b3-2016-04-05-14-14-00.bag
roslaunch cartographer_ros demo_backpack_3d.launch bag_filename:=${HOME}/Downloads/b3-2016-04-05-14-14-00.bag`
```