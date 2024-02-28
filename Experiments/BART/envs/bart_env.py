import numpy as np
import random


class Bart_env():
    """
    This class represents the environment for the Balloon Analogue Risk Task (BART).

    Attributes:
        _max_pumps (list): List of maximum pumps for each balloon.
        _probabilities (list): List of probabilities of each balloon exploding.
        _no_balloons (int): Number of balloons.
        _trials_left (list): List of remaining trials for each balloon.

    Methods:
        __init__(max_pumps: list, trials: int): Initializes the class with the given maximum pumps and number of trials.
        reset_idx(idx): Resets the probability and decreases the number of trials left for the balloon at the given index.
        step(idx, N): Performs a step in the task for the balloon at the given index and returns whether the balloon exploded.
        randomly_sample(): Randomly samples a balloon that still has trials left and returns its index.

    The class is initialized with a list of maximum pumps for each balloon and the number of trials. The probabilities of each balloon exploding are calculated as the inverse of the maximum pumps. The number of trials left for each balloon is initialized to the given number of trials.

    The `reset_idx` method is used to reset the probability of the balloon at the given index after each trial.

    The `step` method is used to perform a step in the task for the balloon at the given index. It updates the probability of the balloon exploding and returns whether the balloon exploded.

    The `randomly_sample` method is used to randomly sample a balloon that still has trials left. It raises an exception if all balloons have 0 trials left.
    """
    def __init__(self, max_pumps: list, trials: int):
        random.shuffle(max_pumps)
        self._max_pumps = max_pumps
        self._probabilities = 1 / np.array(self._max_pumps)
        self._no_balloons = len(self._max_pumps)
        self._trials_left = [trials] * self._no_balloons

    def reset_idx(self, idx):
        """
        This method is used to reset the probability and decrease the number of trials left for the balloon at the given index.
        """
        self._probabilities[idx] = 1 / self._max_pumps[idx]
        self._trials_left[idx] -= 1

    def step(self, idx, N):
        """
        This method is used to perform a step in the task for the balloon at the given index.

        Args:
            idx (int): Index of the balloon.
            N (int): Total number of balloons.
            
        Returns:
            exploded (bool): Whether the balloon exploded.
        """
        exploded = np.random.rand() < self._probabilities[idx]
        if N <= 1:
            self._probabilities[idx] = 1
        else:
            self._probabilities[idx] *=  (N/(N-1))
        return exploded
    
    def randomly_sample(self):
        """
        randomly sample a balloon that does not have 0 trials left
        """
        try:
            idx = np.random.choice(np.where(np.array(self._trials_left) > 0)[0])
        except:
            import ipdb; ipdb.set_trace()
        return idx
    
