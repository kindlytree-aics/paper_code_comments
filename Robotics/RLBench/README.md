    

# RLBench

## 问题1：RLBench和ros中的gym提供的机器人模拟器环境对比
RLBench为机器人强化学习等提供了一个基准平台，特别是单臂机器人的常用操作任务提供模拟仿真环境，而gym是强化学习中更加通用的平台，提供了众多基础的物理仿真环境（如小车平衡杆CartPole-v0等）进行强化学习基础算法实验。
RLBench默认使用的是Franka Panda机械臂，也可以支持其他机器人，有人已经移植了UR5等，仿真环境要和真实的机器人尽可能保持一致性（包括外形等几何结构一致性，动力学运动一致性以及仿真控制操作接口一致性），这样Sim2Real时会保持最大限度的兼容，迁移也更为方便和有效。

补充模拟器相关信息如下：

仿真模拟数据采集生成，这里主要用到了仿真器CoppeliaSim以及PyRep库通过python语言和仿真器之间的接口调用，如调用仿真器启动场景(PyRep的launch接口打开场景描述文件task_design.ttt)和加载物体模型(PyRep的import_model接口加载task_ttms目录下的ttm文件)以及机器人模型（PyRep的import_model接口加载robot_ttms目录下的ttm文件）。

模拟器单独使用的方法：
- 在`~/CoppeliaSim`目录下，通过`./coppeliaSim.sh`启动模拟器软件
- 在菜单中选择`open scene`打开场景文件`rlbench`目录下的`task_design.ttt`场景文件。
  
以及通过菜单`load model`选择`task_ttms`目录下任务相关物体模型文件或者`robot_ttms`下的具体的机器人模型文件，实现将将特定模型加载到场景中。
PyRep库是连接 Python 和 CoppeliaSim 的桥梁，可以使用PyRep库通过python代码加载上面的模型文件，并通过step函数进行渲染，在RLBench正式通过PyPep库的接口实现了和CoppeliaSim模拟环境的集成。

## 问题2：RLBench源代码结构及实现上关键点介绍
RLBench的关键实现由以下几个部分：
1、backend package，定义了几个核心基础类，如Envrionment，robot，scene，task，observation，waypoints等仿真模拟环境基本核心组件或者基础类（如Task为具体的任务的积累定义）
- Observation，主要是视觉数据（多个RGB-D摄像头的非结构化数据）和低维观测数据（如夹爪位置，速度等本体感知观测数据）。如重要的视觉传感器类VisionSensor（摄像头，pyrep.objects.vision_sensor包里面的类）
```
from pyrep.vision_sensor import VisionSensor
sensor = VisionSensor('cam_wrist')

rgb = sensor.capture_rgb()      # (H, W, 3)
depth = sensor.capture_depth()  # (H, W), 单位为米
pc = sensor.pointcloud          # (N, 3)，xyz点云
```

视觉传感器一般会有多个，摄像头布局可以从如下的代码实现中看出，摄像头的数据也是通过模拟器CoppeliaSim种计算得出，根据OpenGL渲染的图像和GPU渲染的深度缓冲区等来源通道获取
```python
self._cam_over_shoulder_left = VisionSensor('cam_over_shoulder_left')
self._cam_over_shoulder_right = VisionSensor('cam_over_shoulder_right')
self._cam_overhead = VisionSensor('cam_overhead')
self._cam_wrist = VisionSensor('cam_wrist')
self._cam_front = VisionSensor('cam_front')
```

| 摄像头名 | 视角类型     | 是否 Eye-in-Hand |
|----------|--------------|------------------|
| `cam_wrist` | 末端执行器视角 | ✅ 是 |
| `cam_over_shoulder_left` | 左肩外部视角 | ❌ 否 |
| `cam_over_shoulder_right` | 右肩外部视角 | ❌ 否 |
| `cam_overhead` | 顶部鸟瞰视角 | ❌ 否 |
| `cam_front` | 正前方视角 | ❌ 否 |

如你需要进一步调整这些摄像头的位置（比如让 wrist 相机更倾斜、更远、更靠近 gripper），可以通过 CoppeliaSim 场景编辑器进行调整，或者在 PyRep 中动态设置 `VisionSensor.set_position()` 和 `set_orientation()`。

2、Scene，场景类，包括观测，机器人，工作区等要素，加载特定的任务，获取观测（`get_observation`）以及一个episode(`get_demo`)等接口  
3、 TaskEnvironment，对具体任务环境进行建模，Environment类对象的get_task来获，并可以通过`get_demos`接口获取更多的函数，
```
    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                try:
                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos
```
4、ActionMode，定义了机器人如何运动，在rlbench的`action_modes`的package里进行的实现，有如下定义示例：

```
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(), # 控制 EEF 目标绝对位姿 (7维: x,y,z,qx,qy,qz,qw)
    gripper_action_mode=Discrete() # 控制夹爪 (1维: 0=关, 1=开)
)
```
ActionMode包含机械臂和夹爪两个部分的运动，如上的示例中先运动机械臂，最后运动夹爪以完成特定的任务。
也有同时运动的类的实现:`JointPositionActionMode`


## 问题3：具体任务的个性化定义如何实现？

RLBench中提供了具体的100多个机器人常用操作任务，基类定义在backend package的Task类，各个任务子类定义在tasks package里，每一个特定任务一个py文件进行定义, 每一个任务类一个py文件，如open_box.py。任务类实现了几个关键的函数
- `init_task`定义了初始化任务相关的物体，如bin， rubbish等，以及注册任务完成的条件
```
self.block = Shape('block')
success_detector = ProximitySensor('success')
self.target = Shape('target')
self.boundary = SpawnBoundary([Shape('boundary')])

success_condition = DetectedCondition(self.block, success_detector)
self.register_success_conditions([success_condition])
```
- `init_episode`初始化,定义一次任务的操作过程的初始化，每一次任务都有随机的一些场景设置，如颜色，位置等
```
     def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self.target_topPlate.set_color([1.0, 0.0, 0.0])
        self.target_wrap.set_color([1.0, 0.0, 0.0])
        self.variation_index = index
        button_color_name, button_rgb = colors[index]
        self.target_button.set_color(button_rgb)
        self.register_success_conditions(
            [ConditionSet([self.goal_condition], True, False)])
        return ['push the %s button' % button_color_name,
                'push down the %s button' % button_color_name,
                'press the button with the %s base' % button_color_name,
                'press the %s button' % button_color_name]
                
```
- `variation_count`函数,定义了一些任务的不同变体的个数，如物体的颜色类别数等
```
def variation_count(self) -> int:
    return len(colors)
```

## 问题4：给出一个基于RLBench实现的机器人操作任务的模拟数据获取过程

### 环境准备和安装

- conda或miniconda，创建虚拟环境 `robotic_env`

```
conda create --name robotic_env python=3.9
conda activate robotic_env
```

- 安装RLBench，参看连接[install](https://github.com/stepjam/RLBench?tab=readme-ov-file#install)

```
pip install gymnasium

sudo apt-get update
sudo apt-get install libgl1-mesa-dri

mkdir -p /usr/lib/dri/
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/

sudo apt-get install --reinstall libgl1-mesa-dri
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 采集模拟数据

RLBench中通过get_demos收集数据，是通常的模拟仿真收集数据的方法, 会调用 PyRep 控制机器人实时执行演示（效率较低），具体的请参考[数据采集代码](./dataset_sim.py)的实现

```
# --- 数据采集 ---
demos = []
print(f"Attempting to collect {NUM_DEMOS_TO_SAVE} demonstrations...")
# get_demos 需要任务支持演示生成，PickAndPlace 支持
# max_attempts: 尝试多少次生成一个有效的演示
# variation_number: 可指定特定变体，-1 表示随机
try:
    #demos = task.get_demos(amount=NUM_DEMOS_TO_SAVE, live_demos=False) # live_demos=True 会实时运行仿真获取
    demos = task.get_demos(amount=NUM_DEMOS_TO_SAVE, live_demos=True) # live_demos=True 会实时运行仿真获取
except Exception as e:
    print(f"Error getting demos: {e}")
    print("Make sure the task supports get_demos and dependencies are met.")

```

