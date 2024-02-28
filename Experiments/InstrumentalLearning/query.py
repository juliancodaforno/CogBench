from torch.distributions import Binomial
import numpy as np
import pandas as pd
from tqdm import tqdm
import gym
import envs
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm

class TwoArmedBandit4Context_ExpForLLM(Experiment):
    """
    This class represents an experiment for the 2-armed bandit task with 4 contexts. It extends the base Experiment class.
    The experiment is designed to be run with an LLM. The LLM is used to generate choices in each trial of the task.

    Attributes:
        get_llm (function): A function that returns an LLM object.
        arms (list): List of engines to use.
        num_runs (int): Number of runs.
        version_number (str): Version number of the experiment.

    Methods:
        add_arguments_(): Adds additional arguments to the parser.
        run_single_experiment(llm): Runs an LLM on a single experiment of the 2-armed bandit task with 4 contexts.

    The class inherits from the Experiment class. It overrides the `__init__` method to add additional arguments to the parser and the `run_single_experiment` method to run the 2-armed bandit task with the given LLM.

    The `run_single_experiment` method simulates a 2-armed bandit task using the given LLM. It generates a prompt for each trial, lets the LLM choose an action, and then steps into the next trial. 
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--arms', nargs='+', default=['F', 'J'], help='List of engines to use')
        self.parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')
        self.arms = self.parser.parse_args().arms

    def run_single_experiment(self, llm):
        """
        Runs an LLM on a single experiment of the 2-armed bandit task with 4 contexts.

        This function simulates a 2-armed bandit task using a given LLM. It generates a prompt for each trial, lets the LLM choose an action, and then steps into the next trial. The results of each trial are stored in a DataFrame.

        Parameters:
        llm (LLM): The LLM object which will be used to generate the choices.

        Returns:
        df (pd.DataFrame): A DataFrame with the results of the experiment. Each row represents a trial and contains the trial number, current machine, mean rewards for each machine, actual rewards for each machine, chosen action, and received reward.
        """
        self.Q_, self.A_ = llm.Q_A      
        llm.random_fct = self.random_fct
        llm.format_answer = "Machine "
        llm.default_query_specifications = '(Give the answer in the form \"Machine <your answer>.\")'
        # Define a function that generates a choice from the LLM and keeps only the choices that are in self.arms
        llm_choice = lambda x: self.keep_arms(llm.generate(x))
        data = []
        done = False
        env = gym.make('palminteri2017-v0')
        env.reset()
        num_trials = env.max_steps
        instructions, history, trials_left, question = self.reset(env, env.contexts[0, 0], env.max_steps)
        current_machine = env.contexts[0, 0]
        # Loop through each trial in the environment and generate a prompt for the LLM to act on. 
        for t in range(num_trials):
            prompt = instructions + trials_left + "\n" + history + "\n"+ question
            # LLM acts
            self.arms = [i for i in env.arms[int(current_machine)-1].keys()]
            action = llm_choice(prompt)
            if action not in env.arms[int(current_machine)-1].keys():
                print(f'Invalid action: {action} not in {env.arms[int(current_machine)-1].keys()}')
                import ipdb; ipdb.set_trace()
                print('---')

            action_int = int(env.arms[int(current_machine)-1][action]) #torch.tensor([int(action)])
            rewards = env.rewards[0, t, action_int].item()
            # save values
            row = [t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), action_int,  rewards]
            data.append(row)
            if not done:
                # step into the next trial
                history, trials_left, current_machine, question, done = self.step(env, history, current_machine, action, t)
            row = [t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), action_int,  rewards]  
            data.append(row)
        df = pd.DataFrame(data, columns=['trial', 'task', 'mean0', 'mean1', 'reward0', 'reward1', 'choice', 'reward'])
        return df

    def rdm_choice_agent(self):
        return self.arms[0] if np.random.rand() < 0.5 else self.arms[1]

    def random_fct(self):
        """Randomly choose between the two arms."""
        return np.random.choice(self.arms)
    
    def keep_arms(self, text):
        '''
        Args:
            text (str): text to only keep arms from
        Returns:
            text (str): text with only arms
        '''
        while text[-1] not in self.arms:
            if len(text) == 0:
                text = text[:-1]
            else:
                # If text is empty, the LLM will choose randomly between the arms (This happens very rarely but avoids problems)
                return self.rdm_choice_agent()
        return text[-1]
    
    def reset(self, env, context, num_trials):
        """
        Resets the environment and context for a new set of trials.

        This function prepares the instructions, history, remaining trials, and question for the user based on the given environment, context, and number of trials.

        Parameters:
        env (Environment): The environment object representing the casinos and slot machines.
        context (int): The current context, representing the current casino.
        num_trials (int): The total number of trials the user will perform.

        Returns:
        tuple: A tuple containing the instructions, history, remaining trials, and question for the user.
        """
        arm1_, arm2_ = env.arms[int(context)-1].keys()
        instructions = f"You are going to visit four different casinos (named 1, 2, 3 and 4) {int(num_trials/4)} times each. Each casino owns two slot machines which all return either 1 or 0 dollars stochastically with different reward probabilities. "
        trials_left = f"Your goal is to maximize the sum of received dollars within all {num_trials} visits.\n"
        history = ""    
        question = f"{self.Q_} You are now in visit {env.t + 1} playing in Casino {int(context)}." \
            f" Which machine do you choose between Machine {arm1_} and Machine {arm2_}?"
        
        return instructions, history, trials_left, question

    def step(self, env, history, prev_machine, action, t):
        """
        Performs a step in the environment based on the user's action and updates the history.

        This function takes an action in the environment, receives the reward and next context, and updates the history and remaining trials. It also prepares the next question for the user.

        Parameters:
        env (Environment): The environment object representing the casinos and slot machines.
        history (str): The history of actions and rewards so far.
        prev_machine (int): The machine that was used in the previous step.
        action (str): The action chosen by the user.
        t (int): The current time step.

        Returns:
        tuple: A tuple containing the updated history, remaining trials, next machine, question for the user, and a boolean indicating whether the episode is done.
        """
        # get reward and next context
        observation, reward, done, _ = env.step(env.arms[int(prev_machine)-1][action])
        next_machine = observation[0, 3]
        
        if t==0:
            history =  "You have received the following amount of dollars when playing in the past: \n"
        # update history based on current action and trials
        history += f"- Machine {action} in Casino {int(prev_machine.numpy())} delivered {float(reward)} dollars.\n"
        
        # update trials left
        trials_left = f"Your goal is to maximize the sum of received dollars within {env.max_steps} visits.\n"
        arm1_, arm2_ = env.arms[int(next_machine)-1].keys()
        if Binomial(1, 0.5).sample() == 1:  #randomly change their order for prompt
            arm1_ , arm2_ = arm2_, arm1_
        question = f"{self.Q_} You are now in visit {env.t+1} playing in Casino {int(next_machine)}." \
            f" Which machine do you choose between Machine {arm1_} and Machine {arm2_}?"
        
        return history, trials_left, next_machine, question, done


        

if __name__ == '__main__':
    experiment = TwoArmedBandit4Context_ExpForLLM(get_llm)
    experiment.run()

    
