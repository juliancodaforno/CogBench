import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm


class TwoStepTaskExpForLLM(Experiment):
    """
    This class represents a two-step task experiment for an LLM. It extends the base Experiment class.

    The class is initialized with a function `get_llm` that is used to get the LLM generate function.

    The class has four main attributes:
    - `arms`: A list of engines to use. Default is ['X', 'Y'].
    - `num_trials`: The number of trials. Default is 20, replicating the setup in (Binz & Schulz, 2023).
    - `num_runs`: The number of runs. Default is 100.
    - `version_number`: The version number of the experiment. Default is '1'.

    These attributes are set using command line arguments when running the script.

    Methods:
    - `__init__(self, get_llm)`: Initializes the class and adds the command line arguments.
    - `add_arguments_(self)`: Adds the command line arguments using argparse.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--arms', nargs='+', default=['X', 'Y'], help='List of engines to use')
        self.parser.add_argument('--num_trials', type=int, default=20, help='Number of trials. Set to 20 by default to replicate (Binz & Schulz, 2023).')
        self.parser.add_argument('--num_runs', type=int, default=100, help='Number of runs')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment')
        parser = self.parser.parse_args()
        self.arms = parser.arms
        self.num_trials = parser.num_trials

    def run_single_experiment(self, llm):
        """Runs an LLM on single experiment of the Two-step task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm_choice = lambda x, arms: self.keep_arms(llm.generate(x), arms)
        action_to_index = {"D": 0, "F": 1, "J": 2, "K": 3}
        reward_probs = np.random.uniform(0.25, 0.75, (4,))
        start_text = "You will travel to foreign planets in search of treasures.\n"\
        "When you visit a planet, you can choose an alien to trade with.\n"\
        "The chance of getting treasures from these aliens changes over time.\n"\
        "Your goal is to maximize the number of received treasures.\n\n"\

        previous_interactions = []
        data = []

        for i in range(self.num_trials):
            total_text = start_text
            if len(previous_interactions) > 0:
                total_text += "Your previous space travels went as follows:\n"
            for count, interaction in enumerate(previous_interactions):
                days = " day" if (len(previous_interactions) - count) == 1 else " days"
                total_text += "- " + str(len(previous_interactions) - count) + days + " ago, "
                total_text += interaction

            total_text += f"\n{Q_} Do you want to take the spaceship to planet X or planet Y?\n"
            llm.format_answer = f"Planet "

            action1 = llm_choice(total_text, arms=('X', 'Y'))
            total_text += " " + action1 + ".\n"
            state = self.transition(action1)

            if state == "X":
                feedback = "You arrive at planet " + state + ".\n"\
                f"{Q_} Do you want to trade with alien D or F?\n"
            elif state == "Y":
                feedback = "You arrive at planet " + state + ".\n"\
                f"{Q_} Do you want to trade with alien J or K?\n"

            total_text +=  feedback
            llm.format_answer = "Alien "

            action2 = llm_choice(total_text, arms=('D', 'F') if state == 'X' else ('J', 'K'))
            treasure = np.random.binomial(1, reward_probs[action_to_index[action2]], 1)[0]

            row = [i, action1, state, action2, treasure, reward_probs[0], reward_probs[1], reward_probs[2], reward_probs[3]]
            data.append(row)

            reward_probs += np.random.normal(0, 0.025, 4)
            reward_probs = np.clip(reward_probs, 0.25, 0.75)

            total_text += " " + action2 + ".\n"
            total_text += "You receive treasures." if treasure else "You receive junk."
            if treasure:
                feedback_item = "you boarded the spaceship to planet " + action1 + ", arrived at planet " + state + ", traded with alien " + action2 + ", and received treasures.\n"
            else:
                feedback_item = "you boarded the spaceship to planet " + action1 + ", arrived at planet " + state + ", traded with alien " + action2 + ", and received junk.\n"
            previous_interactions.append(feedback_item)
        df = pd.DataFrame(data, columns=['trial', 'action1', 'state', 'action2', 'reward', 'probsA', 'probsB', 'probsC', 'probsD'])
        return df


    def transition(self, action):
        if action == "X":
            return np.random.choice(['X', 'Y'], p=[0.7, 0.3])
        else:
            return np.random.choice(['X', 'Y'], p=[0.3, 0.7])
        
    def random_fct(self):
        """If random choice: Coin toss between two arms."""
        return self.arms[0] if np.random.rand() < 0.5 else self.arms[1]

    
    def keep_arms(self, text, arms):
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
    experiment = TwoStepTaskExpForLLM(get_llm)
    experiment.run()
