import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gym
# import envs.bandit
import envs
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm


class MetacognitionExpForLLM(Experiment):
    """
    The `MetacognitionExpForLLM` class extends the base class `Experiment`. It has two main methods: `run_single_experiment` and `add_arguments_`.

    - `run_single_experiment`: This method runs a single experiment of the metacognition task. It uses the LLM to generate the responses (choice and confidence) and the environment to generate the rewards. It then returns a DataFrame with the results of the experiment.
    - `add_arguments_`: This method adds the arguments specific to the metacognition task to the parser.

    The script is designed to be run as a standalone program. When run, it creates an instance of `MetacognitionExpForLLM` and calls the `run` method.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
        self.parser.add_argument('--arms', nargs='+', default=['F', 'J'], help='List of engines to use')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')
        self.arms = self.parser.parse_args().arms

    def set_experiment(self):
        """Sets the experiment depending on the version number
        Args:
            env (gym.Env): Environment of the experiment
        """
        if self.parser.parse_args().version_number == '1':
            mean_rewards = [40, 60]
        elif self.parser.parse_args().version_number == '2':
            mean_rewards = [-10, 10]
        env = gym.make('metacognition-v0', no_blocks=4, mean_rewards=mean_rewards)
        return env
    
    def run_single_experiment(self, llm):
        """Runs an LLM on single experiment of the metacognition task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        letter_to_machine = {self.arms[0]: 0, self.arms[1]: 1}
        llm_choice = lambda x: self.keep_arms(llm.generate(x), self.arms)
        llm_confidence = lambda x: self.del_letters_at_end(llm.generate(x))

        instructions = lambda x: f"{Q_} You are going to a casino that owns two slot machines named machine {x[0]} and {x[1]}.\n"\
            "You earn $ each time you play on one of these machines with one"\
            " machine always having a higher average $ reward. Every 18 to 22 "\
            "trials a switch of block takes place and the other slot machine will now give the "\
            "higher point reward in average. However, you are not told about the change of block. After each choice you have to indicate how confident "\
            "you were about your choice being the best on a scale from 0 to 1. The casino includes 4 blocks of 18 to 22 trials, for a total of 80 trials \'t\'."\
            " Your goal is to interact with both machines and optimize your $ as much as possible by identifying the best machine at a given point in time which comes in hand with being attentive at a potential change of block."\
            " The rewards will range between 20$ and 80$."\

        query = lambda x: f"\n\nYou are now in trial t={env.t+1}. Which machine do you choose between machine {x[0]} and {x[1]}?"
        history = "" 
        env = self.set_experiment()
        env.reset()
        data = []
        # Until 4 blocks have been seen in the experiment
        while not env.done:
            # Give the prompt to the LLM and get the responses
            llm.random_fct = self.rdm_choice_agent #random choice agent if llm is random
            llm.format_answer = f"Machine " 
            llm.default_query_specifications = f" Give the answer in the form 'Machine <your choice>'."
            prompt = instructions(np.random.permutation(self.arms)) + history + query(np.random.permutation(self.arms))
            letter_choice = llm_choice(prompt)   
            llm.default_query_specifications = " Give the answer in the form 'Machine <your choice> with confidence <your confidence>' (where your confidence of your choice being the best is given on a continuous "\
                "scale running from 0 representing "\
                f"\'this was a guess\' to 1 representing \'very certain\' should be given to two decimal places)."
            llm.format_answer = f"Machine {letter_choice} with confidence 0."
            llm.random_fct = self.rdm_confidence_agent #random confidence agent if llm is random
            confidence = llm_confidence(prompt)
            try:
                confidence = float('0.' + confidence)
            except:
                # If confidence is not a number, we will store a confidence of 0. 
                confidence = 0.0
            action = letter_to_machine[letter_choice]
            block_trial = env.block_t
            block_ = env.block
            _, reward, _, _ = env.step(action)
            if env.t == 1:
                history += "\n\nYou have received the following amount of $ when playing in the past: \n"
            history += f"t={env.t}: You chose {letter_choice} with a reported confidence of {confidence}. It rewarded {int(reward)} $.\n" 
            accurate = 1 if action == np.argmax(env.mean_rewards) else 0
            row = [env.t-1, block_, block_trial, action, reward, confidence, accurate]       
            data.append(row)
        df = pd.DataFrame(data, columns=['trial', 'block', 'block_trial', 'choice', 'reward', 'confidence', 'accurate'])
        return df

    def rdm_choice_agent(self):
        """If random choice: Coin toss between two arms."""
        return self.arms[0] if np.random.rand() < 0.5 else self.arms[1]

    def rdm_confidence_agent(self):
        """Returns a random number between 1 and 99, with a leading zero if the number is less than 10."""
        number = np.random.randint(1, 100)
        if number < 10:
            return "0" + str(number)
        else:
            return str(number)
        
    def keep_arms(self, text, arms):
        '''
        Args:
            text (str): text to only keep arms from
        Returns:
            text (str): text with only arms
        '''
        if len(text) == 0:
            # If text is empty, the LLM will choose randomly between the arms
            return np.random.choice(arms)
        while text[-1] not in arms:
            if len(text) > 1:
                text = text[:-1]
            else:
                # If text is empty, the LLM will choose randomly between the arms
                return np.random.choice(arms)
        return text[-1]
    
    def del_letters_at_end(self, text):
        '''
        Args:
            text (str): text to delete letters from end
        Returns:
            text (str): text with letters deleted from end
        '''
        original_text = text
        text = text.replace('Machine F with confidence ', '').replace('Machine J with confidence ', '')
        text = text.replace('0.', '').replace(',', '').replace('.', '').replace(' ', '').replace('\'', '').replace(':"', '').replace('?', '').replace('*', '') #The 0. is because some models use to add a 0. before the confidence or restart the question asked...e.g: 'What is your confidence A: Machine F with confidence 0.'  --> Answer would repeat "Machine F with confidence 0."...
    
        if len(text) == 0:
            print(f'{original_text} for confidence so len(text)=0 so storing nan')
            return np.nan
            # If text is empty, have to choose what to do. For now hits debugger.
        while text[-1].isalpha():
            if len(text) > 1:
                text = text[:-1]
            else:
            # If text is empty, have to choose what to do. For now hits debugger.
                print(f'{original_text} for confidence so storing nan')
                return np.nan

        return text
            
if __name__ == '__main__':
    experiment = MetacognitionExpForLLM(get_llm)
    experiment.run()

    
