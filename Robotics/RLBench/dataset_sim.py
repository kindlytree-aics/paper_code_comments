import os
import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning # 常用且较稳定
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutRubbishInBin

# --- 配置 ---
DATA_SAVE_PATH = 'pick_and_place_demos' # 演示数据保存目录
NUM_DEMOS_TO_SAVE = 2 # 要保存的演示数量
TASK_CLASS = PutRubbishInBin

# --- 环境配置 ---
obs_config = ObservationConfig()
# 选择你需要的观测，这对策略训练至关重要
# ObservationConfig：配置观测空间，决定获取哪些传感器数据。
obs_config.set_all(True) # 先全选，后面根据策略需要筛选
# obs_config.left_shoulder_camera.rgb = True
# obs_config.left_shoulder_camera.depth = False
# obs_config.right_shoulder_camera.rgb = True
# obs_config.wrist_camera.rgb = True
# obs_config.joint_positions = True
# obs_config.gripper_open = True
# obs_config.task_low_dim_state = True # 通常包含目标/物体位置等关键信息

# --- 动作配置 ---
# 使用基于运动规划的末端执行器位姿控制 + 离散夹爪控制
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(), # 控制 EEF 目标绝对位姿 (7维: x,y,z,qx,qy,qz,qw)
    gripper_action_mode=Discrete() # 控制夹爪 (1维: 0=关, 1=开)
)

#     task_directory='/mnt/workspace/opensource/RLBench/rlbench/tasks'
# --- 初始化环境 ---
env = Environment(
    action_mode,
    obs_config=obs_config,
    dataset_root='/mnt/g/dataset/Robotics/rlbench',
    headless=False
    #headless=True # True = 不显示图形界面，更快；False = 显示界面，方便调试
)
env.launch()

# --- 获取任务 ---
# EnvironmentTask
task = env.get_task(TASK_CLASS)

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

print(demos)
# --- 数据处理与保存 ---
if demos:
    #demos = sum(demos, []) # 展平成单个 episode 列表
    print(f"Successfully collected {len(demos)} demonstrations.")

    if not os.path.exists(DATA_SAVE_PATH):
        os.makedirs(DATA_SAVE_PATH)

    for i, demo in enumerate(demos):
        # demo 是一个 Observation 列表
        # 我们需要提取 (observation, action) 对
        # 注意：RLBench 的 demo 结构可能需要适配，通常最后一个 obs 没有对应 action
        demo_data = []
        print(f'trajectory length: { len(demo) }' )
        for j in range(len(demo) - 1):
            obs = demo[j]
            print(f'obs is {obs.__dict__}')
            # 从下一个状态 (demo[j+1]) 的 'misc' 字段中回溯获取导致该状态的动作
            # 这是 RLBench 存储演示动作的常用方式
            action = demo[j+1].misc.get('action', None)
            if action is not None:
                # --- 选择你的策略需要的观测数据 ---
                # 示例：使用手腕相机图像和机器人关节状态
                wrist_rgb = obs.wrist_rgb # (H, W, C) uint8
                joint_pos = obs.joint_positions # (7,) float

                # --- 准备要保存的数据 ---
                # 将图像转换为更适合存储的格式，例如 uint8
                wrist_rgb_save = (wrist_rgb * 255).astype(np.uint8) if wrist_rgb is not None else None

                # 将动作也保存下来 (7维 EEF Pose + 1维 Gripper)
                # 注意： EndEffectorPoseViaPlanning 输出 7 维位姿
                eef_pose = action[:7]
                gripper_action = action[7:] # 通常是 [0.] 或 [1.]

                # 检查数据是否有效
                if wrist_rgb_save is not None and joint_pos is not None:
                     demo_data.append({
                         'wrist_rgb': wrist_rgb_save,
                         'joint_positions': joint_pos.astype(np.float32),
                         'action_eef_pose': eef_pose.astype(np.float32),
                         'action_gripper': gripper_action.astype(np.float32)
                     })

        # 保存处理后的演示数据为 .npz 文件
        if demo_data:
             np.savez_compressed(os.path.join(DATA_SAVE_PATH, f'demo_{i}.npz'), steps=demo_data)
             print(f"Saved demo {i} with {len(demo_data)} steps.")
        else:
             print(f"Warning: Demo {i} had no valid steps after processing.")

else:
    print("No demonstrations were collected.")

# --- 关闭环境 ---
env.shutdown()
print("Environment shut down.")