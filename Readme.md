[开发文档](https://bytedance.feishu.cn/docx/KfY9dfRsdosbI5xf5zCcED4HnOd)  
![equation](https://latex.codecogs.com/png.image?\dpi{200}\int_{-\infty}^{\infty}%20e^{-x^2}%20dx%20=%20\sqrt{\pi})  
+---------------------+
|   RL_ENV_Controller  |
+---------------------+
| -environment: str   |
| -algorithm: str     |
| -parameters: dict   |
| -state: str         |
| -rewards: List[float]|
| -episode_lengths: List[int]|
| -episode: int       |
| -step: int          |
| -train_thread: Thread|
+---------------------+
| +__init__(environment: str, algorithm: str, parameters: dict)|
| +start_training()   |
| +stop_training()    |
| +get_training_state(): str|
| +get_episode_rewards(): List[float]|
| +get_episode_lengths(): List[int]|
| +get_episode(): int  |
| +get_step(): int     |
+---------------------+
