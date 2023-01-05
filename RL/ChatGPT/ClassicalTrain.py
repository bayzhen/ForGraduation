from RL.override.ODQN import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from RL.maze import Maze
import numpy as np
from typing import Optional, Union
from stable_baselines3.common.vec_env import VecNormalize
from gym import spaces
import torch as th


class MyReplayBuffer(ReplayBuffer):
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "cpu",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True, ):
        super(MyReplayBuffer, self).__init__(buffer_size=buffer_size,
                                             observation_space=observation_space,
                                             action_space=action_space,
                                             device="cpu",
                                             n_envs=1,
                                             optimize_memory_usage=False,
                                             handle_timeout_termination=True)
        self.td_error = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gpu_device = "cuda:0"

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def my_sample(self, net, gamma, batch_size: int, env: Optional[VecNormalize] = None):
        with th.no_grad():
            next_observations = th.tensor(self.next_observations).to(self.gpu_device)
            next_q_values = net(next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            rewards = th.tensor(self.rewards).to(self.gpu_device)
            dones = th.tensor(self.dones).to(self.gpu_device)
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
            observations = th.tensor(self.observations).to(self.gpu_device)
            values, indices = th.max(net(observations), dim=1)
            values = values.reshape(-1, 1)
            del next_observations
            del next_q_values
            del rewards
            del dones
            del observations
            del indices
            td_error = values - target_q_values


if __name__ == '__main__':
    # Create the CartPole environment using the make_vec_env function from stable-baselines
    game_env = Maze(use_image=False, is_random=False)
    # Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
    model = DQN("MlpPolicy",
                env=game_env,
                replay_buffer_class=MyReplayBuffer,
                verbose=2,
                tensorboard_log='./DQN/')

    # Train the model for 1000 steps, using the replay buffer to store experiences
    model.learn(total_timesteps=1000000)
