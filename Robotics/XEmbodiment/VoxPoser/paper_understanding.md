# VoxPoser 论文理解

Generating robot trajectories according to L can be very challenging because L may be arbitrarily long-horizon or under-specified (i.e., requires contextual understanding).
语言指令任务可能是长程任务，而且没有充分的进行上下文的描述。
将语言指令进行分解（通过LLM或基于搜索的规划），然后基于子任务进行优化，基于序列的组合从而完成整个的指令任务




By solving this optimization for each sub-task ℓi, we obtain a sequence of robot trajectories that collectively achieve the overall task specified by the instruction L
如 open the top drawer” and its first sub-task “grasp the top drawer handle” (inferred by LLMs)

## 问题1：论文的总体创新点以及实现思路是什么？

回答：解决对于机器人应用中的开放（通用）指令任务，以及开放（通用）物体，实现密集的6自由度的终端执行器的轨迹合成，为日常通用任务机器人的实现提供一种实现思路和方向。
总体思路和技术途径为：通过大语言模型在基于自由开放的任务指令上的任务理解和任务分解的优势，通过视觉语言模型（如开放词汇目标检测）在通用物体等环境感知方面的优势获得可供性和约束性等信息构建3D价值地图，
为智能体构造通用泛化的观测空间（对应强化学习的智能体的observation）提供了实现途径和方法。

技术实现途径主要表现在：
1、不同层级的LMPS（LMP的介绍参考引文[1]）构建开放任务到具体的轨迹合成的逻辑代码生成；
2、通过视觉语言模型（如开放词汇目标检测）在通用物体等环境感知方面的优势获得可供性和约束性等信息构建3D价值地图，为智能体构造了通用泛化的观测空间表示（对应强化学习的智能体的observation）。
3、路径规划直接基于价值地图进行路径搜索和后处理优化以及采用MPC相关技术实现运动控制的参数优化。

```
We achieve this by first observing that LLMs excel at inferring affordances and constraints given a free-form language instruction.

by leveraging their code-writing capabilities, they can interact with a vision-language model
(VLM) to compose 3D value maps to ground the knowledge into the observation space of the agent.

The composed value maps are then used in a model-based planning framework to zero-shot synthesize closed-loop robot trajectories with
robustness to dynamic perturbations.
```

## 问题2：任务如何分解成多个子任务的？
planning a sequence of steps from natural language instructions [16]–[18] without additional model finetuning.
通过大模型辅助编程的能力来执行机器人任务，具体实现时采用多个层级的LMP进行实现，由于LM为通用问答模型，这里为每一类LMP提供了5-19个query，response对作为上下文实现few-shot prompts功能。
层级的划分大概为如下的各个LMP

- Planner LMP
  - Composer LMP
    - affordance map LMP
    - avoidance map LMP
      具体通过Planner的LMP将任务进行分解为多个子任务。如下为一个实例

```
# .\src\prompts\rlbench\planner_prompt.txt

import numpy as np
from env_utils import execute
from perception_utils import parse_query_obj
import action_utils import composer

objects = ['tissue box', 'tissue', 'bowl']
# Query: Can you pass me a tissue and place it next to the bowl?
composer("grasp the tissue")
composer("back to default pose")
composer("move to 10cm to the right of the bowl")
composer("open gripper")
composer("back to default pose")
# done

# .\src\prompts\rlbench\composer_prompt.txt
import numpy as np
from env_utils import execute, reset_to_default_pose
from perception_utils import parse_query_obj
from plan_utils import get_affordance_map, get_avoidance_map, get_velocity_map, get_rotation_map, get_gripper_map

# Query: move ee forward for 10cm.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map(f'a point 10cm in front of {movable.position}')
execute(movable, affordance_map)

# Query: go back to default.
reset_to_default_pose()

# Query: move the gripper behind the bowl, and slow down when near the bowl.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 15cm behind the bowl')
avoidance_map = get_avoidance_map('10cm near the bowl')
velocity_map = get_velocity_map('slow down when near the bowl')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map, velocity_map=velocity_map)

# Query: move to the back side of the table while staying at least 5cm from the blue block.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point on the back side of the table')
avoidance_map = get_avoidance_map('5cm from the blue block')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map)

# Query: move to the top of the plate and face the plate.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm above the plate')
rotation_map = get_rotation_map('face the plate')
execute(movable, affordance_map=affordance_map, rotation_map=rotation_map)
```

## 问题3：在该系统中具体在实现时如何实现不同的LMP的之间的嵌套调用的呢？

参考[Code as Policys的文档](../CodeAsPolicies/README.md)中的问题4描述。
```
.\VoxPoser\src\interfaces.py setup_LMP函数根据配置生成了多个LMP(有low_level lmps和high_level lmps以及skill-level composition)

  # allow LMPs to access other LMPs
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config']]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)

  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name
  )
```

## 问题4：LLM生成的代码如何和VLM进行交互？

系统的感知模块的具体实现根据算法逻辑而异，向外提供特定接口供调用。如特定的pipeline，在文章中的pipeline较为复杂，分为几个步骤：
1、基于开放词汇目标检测算法模型（Open-Vocab Detector）来检测特定物体，文中采用了google提出的OWL-ViT（Open-World Localization via Vision Transformer），该模型可以基于特定目标的语言描述，将图像和文本描述作为输入，输出目标物体的几何信息；
2、然后将物体的boundingbox作为SAM（Segment Anything Model）模型的输入获取物体的mask；
3、通过视觉跟踪算法对mask进行跟踪；
4；跟踪的mask作为RGB-D图像的输入来获取目标物体或（物体的某个部分）的点云信息。
5、感知的点云信息是相对较为稀疏的，经过smooth插值等方式进行稠密化，

以上pipeline在实现时比如接受rgb图像作为输入，输出为根据开放词汇获取的目标的点云信息，整个pipeline可以提供一个接口，接口的调用方式可以写入特定的LMP中，如 `detect`函数：

```
import numpy as np
from perception_utils import detect

objects = ['green block', 'cardboard box']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper
```

## 问题5：动作规划（MotionPlanner）具体做什么事情以及如何实现？依赖哪些关键的信息或数据？
entity of interest（兴趣实体）一般为两种情况（论文中有对应的图示）：
- ee（end effector）末端执行器（此时任务为终端执行器直接作用到目标上，如将lamp打开）；
- 或entity of interest为目标物体或物体的一部分（如将桌子上的抽屉关上，这时抽屉的把手就是entity of interest）。

价值地图构建(Value Map Composition)， 价值地图的结果以(100,100,100,k)形状的张量来表示体素值地图（vox value map），在可供性和约束性地图时k=1，表示cost代价值，在表示旋转地图时，k=4，表示旋转向量。
路径规划基于costmap采用贪心路径搜索优化方法来求解最优路径，costmap将target map和avoid map进行综合，没有采用Dijkstra算法很可能是一个权衡，秉承足够好效率优先的原则。
将当前路径点体素的附近搜索空间中搜索最小的路径成本，选择下一个waypoint，迭代直至目标（成本变为0），具体实现代码在`.\VoxPoser\src\planners.py`中进行的实现。
在基于贪心搜索获取到的初始路径后，在初始路径的基础上进行后处理优化工作，主要有：路径平滑，路径waypoints降采样等。

其他的轨迹参数的优化：如通过rotation maps（如基于物体的点云信息构建表面法向量）构建夹爪的方向参数，gripper maps构建夹爪的“开/合”状态参数，velocity maps构建速度等。

动作规划的动态模型建模：
- 每执行一个动作，会基于最近的观测状态数据进行路径重新规划，以根据实时的最新的更准确的观测来规划下一个动作。
- 在机器人执行时同时会采用控制算法去优化路径，优化的“动作参数”是发送给机器人底层控制系统或直接驱动硬件的指令，常见的有力矩、关节速度、关节加速度等，这些参数最终通过控制电机等物理器件来实现机器人的运动。
- MPC 的核心作用都是利用模型预测未来的状态，并在满足各种约束（关节限位、速度限制、力矩限制、避障等）的前提下，通过优化（如随机采样、序列二次规划等方法）找到能最好地完成任务（如跟踪路径、最小化能耗）的动作参数序列，并滚动执行。