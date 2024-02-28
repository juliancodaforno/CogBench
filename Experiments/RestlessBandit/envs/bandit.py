import numpy as np
import gym
from gym import error, spaces, utils
import torch

class MetacognitionEnv(gym.Env):
    """
    Description:
        A casino owns two slot machines named machine F and J. You earn $ each time you play on one of these machines with one
        machine always having a higher average $ reward. Every 18 to 22 trials a switch of block takes place and the other slot machine will now give the
        higher point reward in average. However, you are not told about the change of block. After each choice you have to indicate how confident
        you were about your choice being the best on a scale from 0 to 1. The casino includes 4 blocks of 18 to 22 trials, for a total of 80 trials 't'.
        Your goal is to interact with both machines and optimize your $ as much as possible by identifying the best machine at a given point in time which comes in hand with being attentive at a potential change of block.
        The rewards will range between 20$ and 80$.
    Source:
        This environment corresponds to the environment of the metacognition task from Ershadmanesh, S., Gholamzadeh, A., Desender, K., and Dayan, P. Meta-cognitive efficiency in learned value-
        based choice. In 2023 Conference on Cognitive Computational Neuroscience, pp. 29–32, 2023. doi: 10.32470/
        CCN.2023.1570-0. URL https://hdl.handle.net/21.11116/0000-000D-5BC7-D. We just reduced the number of trials from 400 to 80 due to context length restrictions.

    Actions:
        0	Machine F
        1	Machine J
    Reward:
        Reward is a normal distribution with mean and variance of the chosen machine (sampled from N(40, 8) for one arm and N(60, 8) for the other).
    Episode Termination:
        The episode ends after 80 trials.
    """
    def __init__(self, num_actions=2, mean_rewards=[40, 60], var_rewards=[8, 8], step_range_per_block=[18, 22], no_blocks=20): # 1 for simulations
        self.num_actions = num_actions
        self.mean_rewards = np.random.permutation(mean_rewards)
        self.var_rewards = var_rewards
        self.steps_per_block = [21, 20, 20, 19]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 1), dtype=np.float32)
        self.no_blocks = no_blocks
        self.t = 0
        self.block = 0
        self.done = False

    def reset(self):
        """
        Reset the environment for the next block.
        """
        self.block_t = 0
        # permute the rewards for the next block
        self.mean_rewards = [self.mean_rewards[1] , self.mean_rewards[0]]

    
    def step(self, action):
        """
        Execute one step within the environment.
        Args:
            action (int): The action to execute.
        Returns:
            tuple: A tuple containing the new state, the reward, whether the block is done, and additional info.
        """
        reward = torch.normal(torch.tensor(self.mean_rewards[action], dtype=torch.float), torch.tensor(self.var_rewards[action], dtype=torch.float))
        reward = np.round(reward.item(), 2)
        self.t += 1
        self.block_t += 1
        block_done = False
        if self.block_t >= self.steps_per_block[self.block]:
            self.block += 1
            self.reset()
            block_done = True
            self.done = True if (self.block >= self.no_blocks) else False
        return 'n/a', reward, block_done, 'n/a'
    
