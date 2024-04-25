import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
from envs.bart_env import Bart_env
import statsmodels.api as sm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm


class RiskExpForLLM(Experiment):
    """
    This class represents an experiment for the Risk task.

    The experiment is designed to be run with an LLM. 

    Attributes:
        get_llm (function): A function that returns an LLM object.
        num_runs (int): Number of runs. Default is 10.
        max_pumps (list): List of max pumps for each balloon. Default is [128, 32, 8].
        trials (int): Number of trials per balloon. Normally 30 but 15 for LLMs due to context length. Default is 15.
        version_number (str): Version number of the experiment. Default is '1'.

    Methods:
        __init__(get_llm): Initializes the class and adds the command line arguments.
        add_arguments_(): Adds additional arguments to the parser.

    The class inherits from the Experiment class. It overrides the `__init__` method to add additional arguments to the parser.
    The `add_arguments_` method adds additional arguments to the parser. These arguments can be used to customize the experiment.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
        self.parser.add_argument('--max_pumps', type=list, default=[128, 32, 8], help='List of max pumps for each balloon')
        self.parser.add_argument('--trials', type=int, default=15, help='Number of trials per balloon. Normally 30 but 15 for LLMs due to context length')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')

    def run_single_experiment(self, llm):
        """Runs an LLM on single experiment of the BART task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm.format_answer = "Option "
        llm_choice = lambda x: self.del_letters_at_end(llm.generate(x))
        trials = self.parser.parse_args().trials

        if self.parser.parse_args().version_number == '1':
            trials = 15
        elif self.parser.parse_args().version_number == '2':
            trials = 30

        env = Bart_env(self.parser.parse_args().max_pumps, trials)
        data = {'trial': [], 'pumps': [], 'reward': [], 'max_pumps': [], 'exploded': [], 'balloon_trial': [], 'balloon_idx': []}

        alphabet = 'ABCDEFGHIJ'[:env._no_balloons]

        if self.parser.parse_args().version_number == '1':
            instructions = f"In this game you will encounter {env._no_balloons} different balloons labeled {', '.join(alphabet[:-1]) + ' and ' + alphabet[-1]}."\
            f" There will be a total of {trials} balloons for each type of balloon. Your goal is to accumulate as many points as possible without popping the balloon."\
            " You will be presented with a balloon and given the option to inflate it or not."\
            " Each inflation increases the balloon's size and potential points, but also carries a risk of the balloon popping. "\
            "Your task is to decide whether to inflate the balloon or not knowing that a successful inflation adds 1 point from that balloon."\
            " Once you decide to stop inflating the balloon, you can no longer earn points "\
            "from that balloon. If the balloon pops before you stop inflating (skip), you will lose all the points accumulated for that balloon. "\
            f"Your final score will be determined by the total number of points earned across all {trials*env._no_balloons} balloons. "\
            "Your goal is to maximize your final score. "
            history = ""

            # Run the experiment for each balloon by looping through the trials and randomly sampling a balloon
            for trial in range(trials*env._no_balloons):
                print(f"Trial {trial}")#TODO: REMOVE THIS WHEN YOU SEE IT, WAS THERE FOR DEBUGGING PURPOSES
                balloon_idx = env.randomly_sample()
                pursuing = True; no_pumps = 0; observations = ""

                while pursuing:
                    actions = np.random.choice([1, 2], size=2, replace=False)
                    prompt = instructions + history + observations + f"{Q_} You are currently with balloon {trial+1} which is a balloon of type {alphabet[balloon_idx]}. "\
                        f"What do you do? (Option {actions[0]} for 'skip' or Option {actions[1]} for 'inflate')"
                    if (trial == 0) & (no_pumps == 0):
                        history += f"\n\n You observed the following previously where the type of balloon is given in parenthesis:"

                    action_taken = llm_choice(prompt)
                    try:
                        action_taken = int(action_taken)
                    except:
                        print(f"Invalid action... \\{action_taken}\\. Please try again.")
                        import ipdb; ipdb.set_trace()
                                    
                    if action_taken == actions[0]:
                        pursuing = False
                        exploded = False
                    elif action_taken == actions[1]:
                        exploded = env.step(balloon_idx, N=env._max_pumps[balloon_idx]-no_pumps)
                        no_pumps += 1
                        if exploded:
                            pursuing = False
                    else:
                        print(f"Invalid action... \\{action_taken}\\. Please try again.")
                        import ipdb; ipdb.set_trace()

                    observations = f"\n -Balloon {trial+1} ({alphabet[balloon_idx]}): You inflated the balloon {no_pumps} times for a total of {no_pumps*1} points."\
                
                # Data storage
                history += f"\n -Balloon {trial+1} ({alphabet[balloon_idx]}): You inflated the balloon {no_pumps} times for a total of {0 if exploded else no_pumps*1} points. It did {'' if exploded else 'not '}explode."
                data['trial'].append(trial)
                data['exploded'].append(exploded)
                data['pumps'].append(no_pumps)
                data['reward'].append(no_pumps*1*(not exploded))
                data['max_pumps'].append(env._max_pumps[balloon_idx])
                data['balloon_trial'].append(trials - env._trials_left[balloon_idx])
                data['balloon_idx'].append(balloon_idx)
                env.reset_idx(balloon_idx)
        
        elif self.parser.parse_args().version_number == '2': 
            instructions = f"In this game you will encounter {env._no_balloons} different candy dispensers labeled {', '.join(alphabet[:-1]) + ' and ' + alphabet[-1]}."\
            f" There will be a total of {trials} candy dispensers for each type. Your goal is to accumulate as many candies as possible without causing the dispenser to explode."\
            " You will be presented with a candy dispenser and given the option to pump candies into a container or not."\
            " Each pump adds one candy to the container but also carries a risk of the dispenser exploding. "\
            "Your task is to decide whether to pump candies or not, knowing that a successful pump adds 1 candy from that dispenser."\
            " Once you decide to stop pumping candies from a dispenser, you can no longer earn candies from that dispenser. "\
            " If the dispenser explodes before you stop pumping, you will lose all the candies accumulated from that dispenser.  "\
            f"Your final score will be determined by the total number of candies earned across all {trials*env._no_balloons} dispensers. "\
            "Your goal is to maximize your final number of candies. "
            history = ""

            # Run the experiment for each balloon by looping through the trials and randomly sampling a balloon
            for trial in range(trials*env._no_balloons):
                balloon_idx = env.randomly_sample()
                pursuing = True; no_pumps = 0; observations = ""

                while pursuing:
                    actions = np.random.choice([1, 2], size=2, replace=False)
                    prompt = instructions + history + observations + f"{Q_} You are currently with dispenser {trial+1} which is a dispenser of type {alphabet[balloon_idx]}. "\
                        f"What do you do? (Option {actions[0]} for 'skip' or Option {actions[1]} for 'pump candies')"
                    if (trial == 0) & (no_pumps == 0):
                        history += f"\n\n You observed the following previously where the type of dispenser is given in parenthesis:"

                    action_taken = llm_choice(prompt)
                    try:
                        action_taken = int(action_taken)
                    except:
                        print(f"Invalid action... \\{action_taken}\\. Please try again.")
                        import ipdb; ipdb.set_trace()
                                    
                    if action_taken == actions[0]:
                        pursuing = False
                        exploded = False
                    elif action_taken == actions[1]:
                        exploded = env.step(balloon_idx, N=env._max_pumps[balloon_idx]-no_pumps)
                        no_pumps += 1
                        if exploded:
                            pursuing = False
                    else:
                        print(f"Invalid action... \\{action_taken}\\. Please try again.")
                        import ipdb; ipdb.set_trace()

                    observations = f"\n -Dispenser {trial+1} ({alphabet[balloon_idx]}): You pumped {no_pumps} times for a total of {no_pumps*1} candy."\
                
                # Data storage
                history += f"\n -Dispenser {trial+1} ({alphabet[balloon_idx]}): You pumped {no_pumps} times for a total of {0 if exploded else no_pumps*1} candies. It did {'' if exploded else 'not '}explode."
                data['trial'].append(trial)
                data['exploded'].append(exploded)
                data['pumps'].append(no_pumps)
                data['reward'].append(no_pumps*1*(not exploded))
                data['max_pumps'].append(env._max_pumps[balloon_idx])
                data['balloon_trial'].append(trials - env._trials_left[balloon_idx])
                data['balloon_idx'].append(balloon_idx)
                env.reset_idx(balloon_idx)
        
        df = pd.DataFrame(data, columns=['trial', 'pumps', 'reward', 'max_pumps', 'exploded', 'balloon_trial', 'balloon_idx'])
        return df

    def random_fct(self):
        """If random choice: Coin toss between two arms."""
        return '1' if np.random.rand() < 0.5 else '2'
    
    def del_letters_at_end(self, text):
        '''
        Args:
            text (str): text to delete letters from end
        Returns:
            text (str): text with letters deleted from end
        '''
        if len(text) == 0:
            print("Empty text so choosing randomly between the arms.")
            return self.random_fct()
        if len(text) > 1:
            text = text.replace('.', '').replace(',', '').replace(' ', '').replace('\n', '').replace(':', '').replace('-', '').replace('(', '').replace(')', '').replace('*', '')
        while text[-1].isalpha():
            if len(text) > 1:
                text = text[:-1]
            else:
                # If text is empty, the LLM will choose randomly between the arms
                return self.random_fct()

        #Get the first digit from text
        while not text[0].isdigit():
            text = text[1:]
        return text

            
if __name__ == '__main__':
    experiment = RiskExpForLLM(get_llm)
    experiment.run()

