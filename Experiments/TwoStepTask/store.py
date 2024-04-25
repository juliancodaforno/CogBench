import pandas as pd
import os, sys
from tqdm import tqdm 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringModelBasednessScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        # self.parser.add_argument('--columns', nargs='+', default=['performance_score1','performance_score1_name','behaviour_score2','behaviour_score2_name','behaviour_score3','behaviour_score3_name'])
        pass
    
    def get_all_scores(self):
        """
        Overides the baseclass function because regression across all runs makes more sense rather than for each run
        """
        args = self.parser.parse_args()
        engines = args.engines

        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
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
            storing_df = self.get_scores(full_df, storing_df, engine, run=0)
            storing_df.to_csv(scores_csv_name, index=False)

    def get_scores(self, df, storing_df, engine, run):
        """Get the scores for the two-armed bandit task.
        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
            engine (str): Name of the engine used
            run (int): Number of the run
        Returns:
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
        """
        # data for regression
        data = {'reward': [], 'stay': [], 'common': []}         
        for run_idx in df.run.unique():
            df_idx = df[df.run == run_idx]
            stay = (df_idx.action1.to_numpy()[1:] == df_idx.action1.to_numpy()[:-1]).astype(float)
            reward = df_idx.reward[:-1]
            common = (df_idx.action1.to_numpy()[:-1] == df_idx.state.to_numpy()[:-1])
            data['reward'].extend(reward)
            data['stay'].extend(stay)
            data['common'].extend(common)

        #Behaviour score 1
        interaction_effect, ci = self.get_interaction_effect(data)
        print(f'interaction effect: {interaction_effect} +-{ci}')

        #Performance score 1
        rewards_mean = df['reward'].mean()
        
        # Add the final score to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = rewards_mean
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1_CI'] = 1.96 * df['reward'].std() / np.sqrt(len(df['reward']))
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = interaction_effect
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] ==run), 'behaviour_score1_CI'] = ci
        else:
            storing_df.loc[len(storing_df)] = [engine, run, rewards_mean, 'rewards mean', 1.96 * df['reward'].std() / np.sqrt(len(df['reward'])),interaction_effect, 'interaction effect', ci]
        return storing_df
    
    def get_interaction_effect(self, data):
        """
        This function calculates the interaction effect between 'reward' and 'common' on 'stay' using OLS regression.

        Parameters:
        data (DataFrame): The input data. It should have columns 'reward', 'common', and 'stay'.

        The function first replaces 0 with -1 in 'reward' and False with -1 in 'common' to avoid issues when calculating the interaction effect. It then creates a new column 'interaction' which is the product of 'reward' and 'common'.

        Returns:
        interaction_effect (float): The coefficient of the interaction term from the regression.
        ci (float): The 95% confidence interval of the interaction effect.
        """
        df = pd.DataFrame(data)
        # OLS regression of x1=reward, x2=common, x3=x1*x2 -> stay (not multi-level because we use different runs for a given LLMs not different participants)) 
        df['reward'] = df['reward'].replace(0, -1)
        df['common'] = df['common'].map({False: -1, True: 1}) #-1 instead of 0 to avoid issues when calculating interaction effect
        df['interaction'] = df['reward'] * df['common']
        formula = 'stay ~ reward + common + interaction'
        model = smf.ols(formula, data=df) # not multi_leveled because runs for a given LLM and not storing across different human participants
        result = model.fit()
        interaction_effect = result.params['interaction']
        ci = result.bse.loc['interaction'] * 1.96
        print(f'interaction effect: {interaction_effect}, ci: {ci}')

        return interaction_effect, ci

if __name__ == '__main__':
    StoringModelBasednessScores().get_all_scores()
