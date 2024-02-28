import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch
import random

class UrnTask:
    def __init__(self, informative_lh): # 1 for simulations        
        '''
        Sets the environment
        args:
            informative_lh: if True, then the likelihood is informative, otherwise the prior is informative
        returns:
            left_prob: the probability of left urn
            red_prob: the probability of red ball !!in the left urn!!
        '''
        self.informative_lh = informative_lh 
        if self.informative_lh:
            self.red_prob = np.random.choice([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) #blue prob is 1-red_prob
            self.left_prob = np.random.choice([0.4, 0.5, 0.5, 0.6]) 
        else:
            self.left_prob = np.random.choice([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) 
            self.red_prob = np.random.choice([0.4, 0.5, 0.5, 0.6]) #blue prob is 1-red_prob

    
    def step(self, left_urn, right_urn):
        """
        This function simulates a step in the urn experiment. It randomly selects an urn based on `self.left_prob` and then draws a ball from the selected urn.

        Parameters:
        left_urn (str): The identifier for the left urn.
        right_urn (str): The identifier for the right urn.

        The function first spins a "wheel of fortune" to decide which urn to draw from. If a random number is less than `self.left_prob`, it selects the left urn and sets `red_prob` to `self.red_prob`. Otherwise, it selects the right urn and sets `red_prob` to `1 - self.red_prob`.

        Then, it draws a ball from the selected urn. If a random number is less than `red_prob`, it observes a red ball. Otherwise, it observes a blue ball.

        Returns:
        obs (str): The color of the observed ball ('red' or 'blue').
        urn (str): The identifier of the urn from which the ball was drawn.
        """

        # Spin the wheel of fortune
        if np.random.rand() < self.left_prob:
            urn = left_urn
            #left urn probabilities
            red_prob = self.red_prob
        else:
            urn =  right_urn
            #right urn probabilities
            red_prob = 1 - self.red_prob

        # Draw a ball
        if np.random.rand() < red_prob:
            obs = 'red'
        else:
            obs = 'blue'
        
        return obs, urn
        



            


