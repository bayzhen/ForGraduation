[开发文档](https://bytedance.feishu.cn/docx/KfY9dfRsdosbI5xf5zCcED4HnOd)  
![equation](https://latex.codecogs.com/png.image?\dpi{200}\int_{-\infty}^{\infty}%20e^{-x^2}%20dx%20=%20\sqrt{\pi})  
1. UReinforcementLearningComponent
    - 作为组件添加到虚幻引擎中的角色上，负责处理强化学习任务的初始化、通信、状态收集等功能。

2. URLAction
    - 表示虚幻引擎中的动作，负责处理强化学习模型输出的动作数据，并将其应用到角色上。

3. URLState
    - 表示虚幻引擎中的状态，负责收集虚幻引擎中的环境状态信息，并将其转换为强化学习模型所需的数据格式。

4. URLReward
    - 表示虚幻引擎中的奖励，负责计算当前环境状态下的奖励值，并将其传递给强化学习模型。

5. URLDone
    - 表示虚幻引擎中的任务结束条件，负责判断当前环境状态是否满足任务结束条件，并将结果传递给强化学习模型。

6. RL_ENV_Controller
    - 控制强化学习环境的整体流程，包括与Python后端的通信、控制虚幻引擎中角色的行为等。
