import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import remove_repeated_sentences, remove_excessive_newlines


class Experiment:
    """Base class to represent a single experiment"""
    def __init__(self, get_llm):
        """Initialize the experiment"""
        self.parser = argparse.ArgumentParser(description="Run experiments.")
        self.get_llm = get_llm
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--engines', nargs='+', default=['random'], help='List of engines to use')
        self.parser.add_argument('--max-tokens', type=int, default=2, help='Maximum number of tokens')
        self.parser.add_argument('--temp', type=int, default=0, help='Temperature for LLM')
        self.parser.add_argument('--debug', action='store_true', default=False, help='Debug mode to print the output of the LLM at each step of the experiment')
        #TODO: Add flag --num_runs and --version number in each subclass

    def run(self):
        """Run the experiments"""
        args = self.parser.parse_args()
        engines = args.engines
        print(f"Running experiment with engines: {engines}, num_runs: {args.num_runs} and max_tokens: {args.max_tokens}")
        for engine in engines:
            print(f'Engine :------------------------------------ {engine} ------------------------------------')
            # Load LLM
            llm = self.get_llm(engine, args.temp, args.max_tokens) 
            llm.debug = args.debug

            # If already some run_files in path, start from the last one
            start_run = 0
            version = 'V' + self.parser.parse_args().version_number
            path = f'./data/{engine}{version if version != "V1" else ""}.csv'
            if os.path.exists(path):
                start_run = pd.read_csv(path).run.max() + 1
                print(f"Starting from run {start_run}")
            
            for run in tqdm(range(start_run, args.num_runs)):
                # Run experiment
                df = self.run_single_experiment(llm)
                #Store data
                df['run'] = run
                self.save_results(df, engine)

    def save_results(self, df, engine):
            """Save the results in a CSV format"""
            # Path name
            version = 'V' + self.parser.parse_args().version_number
            folder = f"data{version if version != 'V1' else ''}"
            os.makedirs(folder, exist_ok=True)
            path = f'./{folder}/{engine}.csv'

            if os.path.exists(path):
                existing_df = pd.read_csv(path)
                existing_columns = existing_df.columns.tolist()

                # Reorder columns in df to match the existing CSV file's columns
                df = df[existing_columns]

            df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    def run_single_experiment(self):
        """Run a single experiment"""
        raise NotImplementedError()

class LLM:
    def __init__(self, llm_info):
        self.llm_info = llm_info
        self.Q_A = ('\n\nQ:', '\n\nA:')
        self.format_answer = "" #Default format of the answer is empty
        self.default_query_specifications = "" #Default query specifications is empty

    def generate(self, text, temp=None, max_tokens=None):
        """ Generate a response from the LLM. 'temp' and 'max_tokens' are made optional to be able to use the same function for all LLMs."""
        if temp is None:
            temp = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        Q_, A_ = self.Q_A    
        # Set the default text for default queries, and alternative text if cot and cb can't be processed
        default_text = text + self.default_query_specifications
        default_text += f"{A_} {self.format_answer}"

        # Append the additional text from prompt engineering techniques
        if self.step_back:
            text += " First step back and think in the following two steps to answer this:"\
                "\nStep 1) Abstract the key concepts and principles relevant to this question in maximum 60 words."\
                "\nStep 2) Use the abstractions to reason through the question in maximum 60 words."\
                    f" Finally, give your final answer in the format 'Final answer: {self.format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                    f"{A_} Step 1)\n"
        elif self.cot:
            text += f" First break down the problem into smaller steps and reason through each step logically in a maximum of 100 words before giving your final answer in the format 'Final answer: {self.format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                f"{A_} Let's think step by step:\n1." 
        else:
            text = default_text
        llm_output = self._generate(text, temp, max_tokens)

        #output processing
        processed_output = self.postprocess(llm_output, text, default_text)

        if self.debug:
            # Help debugging by printing the output of the LLM at each step of the experiment for the user to ensure the LLM is working as expected.
            print(text)
            print(f"\n\nAnswer of LLM:'{processed_output}'")
            import ipdb; ipdb.set_trace()
        return processed_output

    def _generate(self, text, temperature, max_tokens):
        raise NotImplementedError("Subclasses should implement this!")

    def postprocess(self, text, text_input, default_text):
        """
        Postprocess the output of the LLM. This is used to extract the answer from the LLM output in case of CoT and Step_back
        where the answer is given in the format 'Final answer: <answer> after a few sentences.
        Args:
            text (str): The output of the LLM
            text_input (str): The input of the LLM
            default_text (str): The default text is fed to LLM for the simpler standard query if none of the postprocessing techniques for prompt engineering outputs work.
        Returns:
            str: Extracting the answer from the output.
        """
        text = remove_excessive_newlines(text)
        if (self.step_back) or (self.cot):
            # Remove the bold format of the answer from some LLMs
            text = text.replace('**Final answer:**', 'Final answer:')
            text = text.replace('Final answer:\n\n', 'Final answer: ')
            text = text.replace('Final answer:\n', 'Final answer: ')

            # Get rid of multiple "Let's think step by step:" if more than 1 just keep first
            text = text.replace("Let's think step by step:", "")
            #Same idea with repeating Step 1)
            text = text.replace('**Step 1)', '')
            text = text.replace('Step 1)', '')



            # Split the text on "Final answer: "
            parts = text.split(f"Final answer: {self.format_answer}")

            # if answer not in right format, or no answer was given, then generate again
            if (len(parts) == 1) or (len(parts[1].split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0):
                # import ipdb; ipdb.set_trace()
                print(f"Answer not in right format, generating again by appending the format at the end...")
                # First delete possible repetitions of the question which is sometimes the issue
                text = remove_repeated_sentences(text)
                
                # Append the generated answer to the text and generate again by hardcoding the format of the answer conditioned on the previously generated answer.
                # This is handy when the llm does not understand the limit of 100 words and gives a long answer without the final answer. We therefore do a two step generation.
                if self.Q_A[0] in text:
                    # Sometimes it does not give in the format 'Final answer: <answer>', so gives the answer and then generates itself a new question, and therefore here, we get rid of everything after the question.
                    text = text.split(self.Q_A[0])[0]

                # Generate the answer 
                reasoning_text = text.rsplit('\n', 1)[0] + f"\n\nFinal answer: {self.format_answer}"
                new_text = self._generate(text_input + reasoning_text, self.temperature, 5)
                # import ipdb; ipdb.set_trace()
                # print(new_text)

                # If the answer is not in the right format, ask again by:
                # 1. appending the reasoning text and the wrong answer to the input and generating again.
                # 2. intervening, saying that the reasoning was too long. 
                if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0: 
                    print(f"2-Answer still not in right format, something is wrong with the reasoning steps. Generating again by telling the LLM, the response was wrong...")
                    #1. appending the reasoning text and the wrong answer to the input and generating again.
                    wrong_output = new_text[:3]
                    re_ask_question = f"{self.Q_A[0]} I don't understand your final answer, the limit was a maximum of 1{'0' if  self.cot else '2'}0 words. Remember you are a reliable and helpful assistant. Just give me your answer without any more steps in the format 'Final answer: {self.format_answer}<your choice>'.{self.Q_A[1]} Sorry, it was a mistake. Final answer: {self.format_answer}"
                    new_text = self._generate(text_input + reasoning_text + wrong_output + re_ask_question, self.temperature, 5)
                    if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0:
                        print(f"3-Answer still not in right format, something is wrong with the reasoning steps. Generating again by telling the LLM, the response was wrong but without appendix the right format this time because sometimes there is a confusion...")
                        #2. intervening, saying that the reasoning was too long without 
                        new_text = self._generate(text_input + text.rsplit('\n', 1)[0] + re_ask_question, self.temperature, 5)
                        
                        # If can't generate again, then return the default answer
                        if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0:
                            new_text = default_text
                            return self._generate(default_text, self.temperature, 2)[:5]
                parts = ['na', new_text]

            # Return the answer until '.' or '\' or '''
            text = parts[1].split('.')[0]
            text = text.split("\n")[0]
            text = text.split("'")[0]
            text = text[:5]
        return text
        

class RandomLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("Random agent is used!")

    def _generate(self, text, temp, max_tokens):
        return self.random_fct()
    
    def random_fct(self):
        raise NotImplementedError("Should set this random function depending on the task!")
    
class InteractiveLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("Interactive agent is used!")

    def _generate(self, text, temp, max_tokens):
        return input(f"{text}")
        
class StoringScores:
    """Analyze the results of the run"""
    def __init__(self):
        self.columns = ['engine','run','performance_score1','performance_score1_name','behaviour_score1','behaviour_score1_name']
        self.parser = argparse.ArgumentParser(description="Get behavioral and performance scores of experiments and store them.")
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--engines', nargs='+', default=['all'], help='List of engines to use')
        self.parser.add_argument('--no_perf_score', nargs='+',  default=[], help='Number of performance scores')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment. This is used to store the results of the experiment with have some changes seperately')
    def get_all_scores(self):
        args = self.parser.parse_args()
        engines = args.engines
        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
        # self.columns += args.columns
        data_folder = f'data{"V"+args.version_number if args.version_number != "1" else ""}'
        if 'all' in args.engines:
            # if the csv file in format ./data/{engine}.csv exists then add to the list of engines
            engines = [os.path.splitext(file)[0] for file in os.listdir(data_folder)]
        # Check if scores_data exists, else, add the column names 
        storing_df =  pd.read_csv(scores_csv_name) if os.path.isfile(scores_csv_name) else pd.DataFrame(columns=self.columns)
             
        # Loop across engines and runs and store the scores
        for engine in tqdm(engines):
            print(f'Fitting for engine: {engine}-------------------------------------------')
            path = f'{data_folder}/{engine}.csv'
            full_df = pd.read_csv(path)
            no_runs = full_df['run'].max() + 1 
            for run in range(no_runs):
                df_run = full_df[full_df['run'] == run].reset_index(drop=True)
                storing_df = self.get_scores(df_run, storing_df, engine, run)
                storing_df.to_csv(scores_csv_name, index=False)




