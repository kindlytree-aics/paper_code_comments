# VoxPoser 论文理解

Generating robot trajectories according to L can be very challenging because L may be arbitrarily long-horizon or under-specified (i.e., requires contextual understanding).
语言指令任务可能是长程任务，而且没有充分的进行上下文的描述。
将语言指令进行分解（通过LLM或基于搜索的规划），然后基于子任务进行优化，基于序列的组合从而完成整个的指令任务

By solving this optimization for each sub-task ℓi, we obtain a sequence of robot trajectories that collectively achieve the overall task specified by the instruction L
如 open the top drawer” and its first sub-task “grasp the top drawer handle” (inferred by LLMs)

问题1：任务如何分解中多个子任务的？

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
objects = ['tissue box', 'tissue', 'bowl']
# Query: Can you pass me a tissue and place it next to the bowl?
composer("grasp the tissue")
composer("back to default pose")
composer("move to 10cm to the right of the bowl")
composer("open gripper")
composer("back to default pose")
# done
```

问题2：LLM生成的代码如何和VLM进行交互，代码基于什么sdk，是通用的机器人sdk吗？代码和VLM交互的方式具体是怎样的？是代码将作为

LMP生成的中间控制逻辑（Python）

如何将 RLBench 接入语言模型

微调 or Few-shot 提示的两种路线

(A) Few-shot Prompting：推荐入门使用
给模型几个任务描述 + 代码实例，用作上下文提示。例如：

Instruction: Pick up the red block and place it in the green box.
Program:
block_pos = detect_object("red block")
box_pos = detect_object("green box")
pick(block_pos)
place(box_pos)

Instruction: Stack the blue block on the red block.
Program:
...

(B) 微调（Fine-tuning）：适合大规模部署
你可以：

收集一批语言描述 + 对应中间控制程序的样本数据

使用 GPT、Codex 或 Code Llama 等模型进行微调

微调目标是 更好理解动作API、机器人空间、时序结构

📦 HuggingFace + LoRA 是很常用的组合。

“可供性映射（Affordance Mapping）” —— 正统翻译，学术风格

Affordance	可供性 / 可操作性	原教科书翻译，HCI/机器人通用术语

## 问题2：动作规划（MotionPlanner）具体做什么事情以及如何实现 ？

PathPlanner

## 问题3：
