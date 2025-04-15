# VoxPoser 论文理解

Generating robot trajectories according to L can be very challenging because L may be arbitrarily long-horizon or under-specified (i.e., requires contextual understanding).
语言指令任务可能是长程任务，而且没有充分的进行上下文的描述。
将语言指令进行分解（通过LLM或基于搜索的规划），然后基于子任务进行优化，基于序列的组合从而完成整个的指令任务

By solving this optimization for each sub-task ℓi, we obtain a sequence of robot trajectories that collectively achieve the overall task specified by the instruction L
如 open the top drawer” and its first sub-task “grasp the top drawer handle” (inferred by LLMs)

## 问题1：任务如何分解成多个子任务的？

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

## 问题2：在该系统中具体在实现时如何实现不同的LMP的之间的嵌套调用的呢？

python有内置的 `exec`函数可供调用，通过LMP生成的代码字符串直接可以作为 `exec`的参数，从而执行这些代码的逻辑。如果生成的代码中有调用的自定义函数的实现是在其他的LMP中，如子函数，则

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

## 问题3：LLM生成的代码如何和VLM进行交互？

系统的感知模块的具体实现根据算法逻辑而异，如特定的pipeline，在文章中的pipeline较为复杂，分为几个步骤：
1、基于开放词汇目标检测算法模型（Open-Vocab Detector）来检测特定物体，文中					采用了google提出的OWL-ViT（Open-World Localization via Vision Transformer），该模型可以基于特定目标的语言描述，将图像和文本描述作为输入，输出目标物体的几何信息；
2、然后将物体的boundingbox作为SAM（Segment Anything Model）模型的输入获取物体的mask；
3、通过视觉跟踪算法对mask进行跟踪；
4；跟踪的mask作为RGB-D图像的输入来获取目标物体或（物体的某个部分）的点云信息。

以上pipeline在实现时比如接受rgb图像作为输入，输出为根据开放词汇获取的目标的点云信息，整个pipeline可以提供一个接口，接口的调用方式可以写入特定的LMP中，如 `detect`函数：

```
import numpy as np
from perception_utils import detect

objects = ['green block', 'cardboard box']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper
```

## 问题2：动作规划（MotionPlanner）具体做什么事情以及如何实现 ？

PathPlanner

## 问题3：
