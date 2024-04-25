import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
# import envs.bandits
import statsmodels.api as sm
import envs
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm


class ExplorationExpForLLM(Experiment):
    """
    Experiment class for the Horizon task.

    The `ExplorationExpForLLM` class extends the base class `Experiment`. It has two main methods: `run_single_experiment` and `add_arguments_`.

    - `run_single_experiment`: This method runs a single experiment of the Horizon task. It uses the LLM to generate the responses (choice and confidence) and the environment to generate the rewards. It then returns a DataFrame with the results of the experiment.
    - `add_arguments_`: This method adds the arguments specific to the Horizon task to the parser.

    The script is designed to be run as a standalone program. When run, it creates an instance of `ExplorationExpForLLM` and calls the `run` method.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--num_runs', type=int, default=1000, help='Number of runs')
        self.parser.add_argument('--arms', nargs='+', default=['F', 'J'], help='List of engines to use')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')
        self.arms = self.parser.parse_args().arms

    def run_single_experiment(self, llm):
        """Runs an LLM on single experiment of the Horizon task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        self.Q_, self.A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm.default_query_specifications = "(Give your answe in the format 'Machine <your choice>')\n"
        llm.format_answer = "Machine "
        llm_choice = lambda x: self.keep_arms(llm.generate(x))

        actions = [None, None, None, None]
        env = gym.make('wilson2014horizon-v0')
        env.reset()

        instructions, history, trials_left, question = self.reset(env.action, env.rewards)

        for t in range(env.rewards.shape[0] - 4):
            prompt = instructions + history + "\n" + trials_left + question
            # print(prompt)
            action = llm_choice(prompt)
            action = action[0] #only take the first letter
            if action == 'F':
                action_to_append = 0
            elif action == 'J':
                action_to_append = 1
            else:
                action_to_append = None
            actions.append(action_to_append)
            history, trials_left = self.step(history, action, env.rewards, t, env)
            if prompt is None:
                break
        if prompt is not None:
            data = []
            for trial in range(env.rewards.shape[0]):
                action = actions[trial] if trial >= 4 else env.action[trial, 0].item()
                row = [trial, env.mean_reward[0, 0].item(), env.mean_reward[0, 1].item(), env.rewards[trial, 0, 0].item(),  env.rewards[trial, 0, 1].item(), action, env.rewards[trial, 0, action].item()]
                data.append(row)
            df = pd.DataFrame(data, columns=['trial', 'mean0', 'mean1', 'reward0', 'reward1', 'choice', 'reward'])
            return df

    def random_fct(self):
        """If random choice: Coin toss between two arms."""
        return self.arms[0] if np.random.rand() < 0.5 else self.arms[1]
    
    def step(self, history, action, rewards, t, env):
        """
        The step function for the Horizon task.
        Args:
            history (str): The history of the experiment.
            action (str): The action taken by the LLM.
            rewards (torch.Tensor): The rewards of the experiment.
            t (int): The current time step.
            env (gym.Env): The environment.
        Returns:
            history (str): The updated history of the experiment.
            trials_left_string (str): The updated string for the remaining trials.
        """
        num2words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
        trials_left = env.rewards.shape[0] - 5 - t

        if trials_left > 1:
            trials_left = num2words[trials_left] + " additional rounds"
        else:
            trials_left = num2words[trials_left] + " additional round"

        trials_left_string = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"

        if action == "F":
            trial_reward = int(rewards[t + 4, 0, 0].item())
        elif action == "J":
            trial_reward = int(rewards[t + 4, 0, 1].item())
        else:
            return None

        history += "- Machine " + action + " delivered " + str(trial_reward) + " dollars.\n"

        return history, trials_left_string

    def reset(self, actions, rewards):
        """
        The reset function for the Horizon task.
        Args:
            actions (torch.Tensor): The actions of the environment.
            rewards (torch.Tensor): The rewards of the environment.
        Returns:
            instructions (str): The instructions for the experiment.
            history (str): The history of the experiment.
            trials_left (str): The string for the remaining trials.
            question (str): The question for the LLM.
        """
        played_machines = []
        observed_rewards = []
        for t in range(4):
            played_machines.append('J' if actions[t, 0] else 'F')
            observed_rewards.append(str(int(rewards[t, 0, 1].item())) if actions[t, 0] else str(int(rewards[t, 0, 0].item())))

        trials_left = 'one additional round' if rewards.shape[0] == 5 else 'six additional rounds'

        instructions = "You are going to a casino that owns two slot machines.\n"\
            "You earn money each time you play on one of these machines.\n\n"\
            "You have received the following amount of dollars when playing in the past: \n"

        history = "- Machine " + played_machines[0] + " delivered " + observed_rewards[0] + " dollars.\n"\
            "- Machine " + played_machines[1] + " delivered " + observed_rewards[1] + " dollars.\n"\
            "- Machine " + played_machines[2] + " delivered " + observed_rewards[2] + " dollars.\n"\
            "- Machine " + played_machines[3] + " delivered " + observed_rewards[3] + " dollars.\n"

        trials_left = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"
        question = f"{self.Q_} Which machine do you choose?"

        return instructions, history, trials_left, question
        
    def keep_arms(self, text, arms=('F', 'J')):
        '''
        Args:
            text (str): text to only keep arms from
        Returns:
            text (str): text with only arms
        '''
        while text[-1] not in arms:
            if len(text) > 1:
                text = text[:-1]
            else:
                # If text is empty, the LLM will choose randomly between the arms
                return np.random.choice(arms)
        return text[-1]
            
if __name__ == '__main__':
    experiment = ExplorationExpForLLM(get_llm)
    experiment.run()
    
