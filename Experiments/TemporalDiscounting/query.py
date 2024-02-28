import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
import statsmodels.api as sm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm


class TemporalDiscuountingExpForLLM(Experiment):
    """
    This class represents an experiment for the Temporal Discounting task. It extends the base Experiment class.
    The experiment is designed to be run with an LLM. The LLM is used to generate choices in each trial of the task.

    Attributes:
        get_llm (function): A function that returns an LLM object.

    Methods:
        __init__(get_llm): Initializes the class and adds the command line arguments.
        add_arguments_(): Adds additional arguments to the parser.
        run_single_experiment(llm): Runs an LLM on a single experiment of the Temporal Discounting task.

    The class inherits from the Experiment class. It overrides the `__init__` method to add additional arguments to the parser and the `run_single_experiment` method to run the Temporal Discounting task with the given LLM.
    The `run_single_experiment` method simulates a Temporal Discounting task using the given LLM. It generates a prompt for each trial, lets the LLM choose an action, and then steps into the next trial. The results of each trial are stored in a DataFrame.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--num_runs', type=int, default=1, help='Number of runs') # Number of runs is set to 1 as the experiment is the only one not procedurally generated.
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')
        self.parser.add_argument('--base_pay', type=int, default=500, help='Base pay for the experiment. Default is 500 representing human experiment in the US.')
    
    def run_single_experiment(self, llm):
        """Runs an LLM on temporal discounting task. A high score means the agent discounts the future more. E.g: Prefers 500 dollars now over 750 dollars in 12 months for max score.
        Args:
            llm (LLM): LLM object which will be used to generate the responses.
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm.format_answer = "I prefer option "
        llm_choice = lambda x: self.del_letters_at_end(llm.generate(x))
        base_pay = self.parser.parse_args().base_pay
        anomalies = {'present bias':1, 'subaddictivity': 0, 'delay-speedup asymmetry-1':0, 'delay-length asymmetry-2':0}

        # Try to run the experiment, else llm_choice is not working so return nan
        try:
            # temporal discounting1
            prompt = lambda pay_options: f"{Q_} What do you prefer between the following two options:\n - Option 1: Receive {pay_options[0]} dollars now.\n - Option 2: Receive {pay_options[1]} dollars in 12 months."
            score, max_value_factor = self.temporal_discounting(base_pay, prompt, llm_choice, get_max_value_factor=True)
            score += self.temporal_discounting(base_pay*10, prompt, llm_choice)
            # gain-loss asymmetry
            prompt = lambda pay_options: f"{Q_} What do you prefer between the following two options:\n - Option 1: Pay {pay_options[0]} dollars now.\n - Option 2: Pay {pay_options[1]} dollars in 12 months."
            score += self.temporal_discounting(base_pay, prompt, llm_choice)

            if max_value_factor == 'no max value at which agent chose the delayed option?':
                #No anomalies checking required as the agent always chose the delayed option
                print('no max value at which agent chose the delayed option?')
                anomalies["score"] = score
                return pd.DataFrame(anomalies, index=[0])

            # present bias
            prompt = f"{Q_} What do you prefer between the following two options:\n - Option 1: Receive {base_pay} dollars in 12 months.\n - Option 2: Receive {int(base_pay*max_value_factor)} dollars in 24 months."
            option = llm_choice(prompt)
            if option == "1":
                score += 1
            elif option == "2":
                anomalies["present bias"] = 1

            # subaddictivity
            prompt = f"{Q_} What do you prefer between the following two options:\n - Option 1: Receive {base_pay} dollars now.\n - Option 2: Receive {int(base_pay*((max_value_factor-1)*2+1))} dollars in 24 months."
            option = llm_choice(prompt)
            if option == "1":
                score += 1
            elif option == "2":
                anomalies["subaddictivity"] = 1

            # delay-speedup asymmetry
            prompt =  f"{Q_} What do you prefer between the following two options:\n - Option 1: Receive {base_pay} dollars now.\n - Option 2: Wait 12 months for the {base_pay} dollars but with an additional {int(base_pay*(max_value_factor-1))} dollars."
            option = llm_choice(prompt)
            if option == "1":
                score += 1
            elif option == "2":
                anomalies["delay-speedup asymmetry-1"] = 1

            # delay-length asymmetry2
            prompt =  f"{Q_} What do you prefer between the following two options:\n - Option 1: Wait 12 months to receive {int(base_pay*max_value_factor)} dollars now.\n - Option 2: Pay {int(base_pay*(max_value_factor-1))} dollars and receive the {int(base_pay*max_value_factor)} dollars gain now."
            option = llm_choice(prompt)
            if option == "1":
                score += 1
            elif option == "2":
                anomalies["delay-length asymmetry-2"] = 1

            #Add score to anomalies to make it a dataframe
            anomalies["score"] = score
            df = pd.DataFrame(anomalies, index=[0])
            return df
        except:
            print('ERROR: llm_choice is not working')
            return pd.DataFrame({'score':np.nan, 'present bias':np.nan, 'subaddictivity':np.nan, 'delay-speedup asymmetry-1':np.nan, 'delay-length asymmetry-2':np.nan}, index=[0])

    def temporal_discounting(self, base_pay, temporal_discounting_prompt, llm_choice, get_max_value_factor = 0):
        """
        This function simulates a temporal discounting task using an LLM.

        Args:
            base_pay (int): The base pay for the task.
            temporal_discounting_prompt (function): A function that generates a prompt for the task.
            llm_choice (function): A function that uses the LLM to make a choice based on the prompt.
            get_max_value_factor (int): If set to 1, the function also returns the maximum value factor. Default is 0.

        Returns:
            score (int): The score of the LLM on the task.
            max_value_factor (float): The maximum value factor at which the LLM chose the delayed option. Only returned if get_max_value_factor is set to 1.

        The function first generates a prompt and uses the LLM to make a choice. If the LLM chooses the immediate option, the function increases the delayed reward and repeats the process. If the LLM chooses the delayed option, the function decreases the delayed reward and repeats the process. The function keeps track of the score and the maximum value factor at which the LLM chose the delayed option.
        If the LLM's response is not 1 or 2, the function raises a ValueError.
        The function returns the score and, if get_max_value_factor is set to 1, the maximum value factor.
        """
        prompt = temporal_discounting_prompt((base_pay, int(base_pay*1.1)))
        response = llm_choice(prompt)
        if response == "1":
            score = 3
            max_value_factor = 1.2
            prompt =  temporal_discounting_prompt((base_pay, int(base_pay*1.2)))
            response2 = llm_choice(prompt)
            if response2 == "1":
                max_value_factor = 1.5
                score += 1
                prompt =  temporal_discounting_prompt((base_pay, int(base_pay*1.5)))
                response3 = llm_choice(prompt)
                if response3 == "1":
                    score += 1
                    max_value_factor = 'no max value at which agent chose the delayed option?'

        elif response == "2":
            prompt =  temporal_discounting_prompt((base_pay, int(base_pay*1.02)))    
            response2 = llm_choice(prompt)
            score = 2
            max_value_factor = 1.1
            if response2 == "2":
                score -= 1
                max_value_factor = 1.02
                prompt =  temporal_discounting_prompt((base_pay, int(base_pay*1.01)))
                response3 = llm_choice(prompt)
                if response3 == "2":
                    score -= 1
                    max_value_factor = 1.01
        else:
            print('ERROR: response not 1 or 2')
            raise ValueError
        if get_max_value_factor:
            return score, max_value_factor 
        return score

    def random_fct(self):
        """Returns either 1 or 2"""
        return np.random.choice(['1', '2'])
        
    def del_letters_at_end(self, text):
        '''
        Args:
            text (str): text to delete letters from end
        Returns:
            text (str): text with letters deleted from end
        '''
        original_text = text
        text = text.replace(',', '').replace('.', '').replace(' ', '').replace("\\", "").replace(':', '').replace('?', '').replace('*', '')
        if len(text) == 0:
            print(f'{original_text} as option choice so storing random choice')
            return self.random_fct()
            # If text is empty, have to choose what to do. For now hits debugger.
        while text[-1].isalpha():
            if len(text) > 1:
                text = text[:-1]
            else:
            # If text is empty, have to choose what to do. For now hits debugger.
                print(f'{original_text} as option so storing nan')
                return np.nan

        return text

if __name__ == '__main__':
    TemporalDiscuountingExpForLLM(get_llm).run()
    
