import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch
import random

class OptimismBiasTaskPalminteri(gym.Env):
    """
    A class used to represent the Optimism Bias Task Palminteri environment.

    ...

    Attributes
    ----------
    num_actions : int
        The number of actions that can be taken in the environment.
    num_contexts : int
        The number of contexts in the environment.
    batch_size : int
        The size of the batch to be processed.
    max_steps_per_context : int
        The maximum number of steps that can be taken in each context.
    reward_scaling : int
        The scaling factor for the rewards.
    action_space : gym.spaces.Discrete
        The action space of the environment.
    probs : dict
        The probabilities for the 'low' and 'high' rewards.
    observation_space : gym.spaces.Box
        The observation space of the environment.

    Methods
    -------
    reset():
        Resets the environment to its initial state.
    get_observation(last_reward, last_action, time_step, cue_context):
        Returns the current observation given the last reward, last action, current time step, and cue context.
    sample_contextual_rewards(context):
        Samples the rewards for a given context.
    step(action):
        Performs an action in the environment and returns the new observation, reward, done flag, and info dictionary.
    sample_alphabet(alphabet):
        Samples two letters from the given alphabet and returns them along with the remaining alphabet.
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, num_actions=2, max_steps_per_context=2, num_contexts=4, reward_scaling=1, batch_size=1): # 1 for simulations
        
        self.num_actions = num_actions
        self.num_contexts = num_contexts
        self.batch_size = batch_size
        self.max_steps_per_context =  max_steps_per_context 
        self.reward_scaling = reward_scaling
        self.action_space = spaces.Discrete(self.num_actions)
        self.probs =  {'low':0.25, 'high':0.75} #{'low':0., 'high':1.} 
        self.observation_space = spaces.Box(np.ones(5), np.ones(5)) #  5 = action+reward+context+trial

 
    def reset(self):
        """
        Resets the environment to its initial state.

        Returns
        -------
        observation : torch.Tensor
            The initial observation of the environment.
        """
        # keep track of time-steps
        self.t = 0
        self.max_steps = self.max_steps_per_context * self.num_contexts
        
        # sample the letters for each context
        self.arms = []
        self.alphabet = 'ABCDEFGHJKLMNOPQRSTVW'
        for i in range(1, 1+self.num_contexts):
            arm1, arm2, self.alphabet = self.sample_alphabet(self.alphabet)
            self.arms.append({ arm1:0, arm2:1})

        # generate reward functions
        mean_rewards_context_ll, rewards_context_ll = self.sample_contextual_rewards(context=['low', 'low'])
        mean_rewards_context_lh, rewards_context_lh = self.sample_contextual_rewards(context=['low', 'high'])
        mean_rewards_context_hl, rewards_context_hl = self.sample_contextual_rewards(context=['high', 'low']) #['high', 'low'])
        mean_rewards_context_hh, rewards_context_hh = self.sample_contextual_rewards(context=['high', 'high']) #['high', 'high'])

       # integer machines
        self.cue_context = torch.zeros(self.batch_size, self.num_contexts, self.max_steps_per_context, 1)
        self.cue_context[:, 0] = 1
        self.cue_context[:, 1] = 2
        self.cue_context[:, 2] = 3
        self.cue_context[:, 3] = 4

        # stack all the rewards together
        # shape: batch_size X max_steps X num_actions/context_size
        self.mean_rewards = torch.stack((mean_rewards_context_ll, mean_rewards_context_lh, mean_rewards_context_hl, mean_rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
        self.rewards = torch.stack((rewards_context_ll, rewards_context_lh, rewards_context_hl, rewards_context_hh), dim=1).reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, self.num_actions)
        self.contexts = self.cue_context.reshape(self.batch_size, self.max_steps_per_context*self.num_contexts, 1)
        
        # shuffle the orders
        shuffle_idx = torch.randperm(self.max_steps_per_context*self.num_contexts)
        self.mean_rewards =  self.mean_rewards[:, shuffle_idx]
        self.rewards =  self.rewards[:, shuffle_idx] 
        self.contexts = self.contexts[:, shuffle_idx]
        last_reward = torch.zeros(self.batch_size)
        last_action = 0 #torch.zeros(self.batch_size)

        return self.get_observation(last_reward, last_action, self.t, self.contexts[:, 0])

    def get_observation(self, last_reward, last_action, time_step, cue_context):
        """
        Returns the current observation given the last reward, last action, current time step, and cue context.

        Parameters
        ----------
        last_reward : float
            The reward from the last action.
        last_action : int
            The last action that was taken.
        time_step : int
            The current time step.
        cue_context : torch.Tensor
            The current cue context.

        Returns
        -------
        observation : torch.Tensor
            The current observation of the environment.
        """
        return torch.cat([
            torch.tensor([last_reward]).unsqueeze(-1), #last_reward.unsqueeze(-1),
            torch.tensor([last_action]).unsqueeze(-1),
            torch.ones(self.batch_size, 1) * time_step,
            cue_context],
            dim=1)

    def sample_contextual_rewards(self, context):
        """
        Samples the rewards for a given context.

        Parameters
        ----------
        context : list
            The context for which to sample the rewards.

        Returns
        -------
        mean_rewards_context : torch.Tensor
            The mean rewards for the given context.
        rewards_context : torch.Tensor
            The sampled rewards for the given context.
        """
        assert len(context)==self.num_actions, "lengths of context and actions do not match"
        ones = torch.ones((self.batch_size, self.max_steps_per_context, self.num_actions))
        mean_rewards_context = torch.zeros((self.batch_size, self.max_steps_per_context, self.num_actions))
        for idx, option in enumerate(context):
            ones[..., idx] = ones[..., idx]*self.probs[option]
            mean_rewards_context[..., idx] = self.probs[option]
        rewards_context = torch.bernoulli(ones)
        return mean_rewards_context, rewards_context

    def step(self, action):
        '''
        Performs an action in the environment and returns the new observation, reward, done flag, and info dictionary.

        Parameters
        ----------
        action : int
            The action to be performed.

        Returns
        -------
        observation : torch.Tensor
            The new observation after performing the action.
        reward : float
            The reward for performing the action.
        done : bool
            A flag indicating whether the episode is done.
        info : dict
            A dictionary containing additional information, such as the regret.
        '''
        regrets = self.mean_rewards[:, self.t].max(1).values[0] - self.mean_rewards[:, self.t][:, action][0]
        reward = self.rewards[:, self.t][:, action][0]
        reward = reward / self.reward_scaling
        self.t += 1
        done = True if (self.t >= self.max_steps-1) else False
        
        observation = self.get_observation(reward, action, self.t, self.contexts[:, self.t])
        return observation, reward, done, {'regrets': regrets.mean()}
    
    def sample_alphabet(self, alphabet): 
        arm1 = random.choice(alphabet)
        alphabet = alphabet.replace(arm1, '')
        arm2 = random.choice(alphabet)
        alphabet = alphabet.replace(arm2, '')
        return arm1, arm2, alphabet
