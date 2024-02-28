import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm 
import statsmodels.api as sm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringExplorationScores(StoringScores):
    """
    The class `StoringExplorationScores` extends a base class `StoringScores`. 
    The `StoringExplorationScores` class is used to store and manage scores from experiments. It has two main methods: `get_all_scores` and `get_scores`.

    - `get_all_scores`: This method retrieves all the scores from the experiments. It first checks if a file named `scores_data.csv` exists. If it does, it reads the data from it into a DataFrame. If it doesn't, it creates a new DataFrame with column names specified in `self.columns`. It then loops over all the engines specified in the command line arguments, reads the data for each engine, and calculates the scores for each participant in the experiment. The scores are then stored back into the `scores_data.csv` file.
    - `get_scores`: This method calculates the scores for a single participant in an experiment. It performs a logistic regression on the data to calculate the scores for two conditions: 'equal' and 'unequal'. The scores are then added to the `storing_df` DataFrame.

    The script is designed to be run as a standalone program. When run, it creates an instance of `StoringExplorationScores` and calls the `get_all_scores` method.
    """
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['behaviour_score2','behaviour_score2_name'])

    def get_all_scores(self):
            '''
            Overides the baseclass one because contrary to most experiments, each 100 runs will count as one "run"
            '''
            args = self.parser.parse_args()
            engines = args.engines

            if 'all' in args.engines:
                # if the csv file in format ./data/{engine}.csv exists then add to the list of engines
                engines = [file.split('.')[0] for file in os.listdir('./data') if os.path.isfile(f'./data/{file}') and file.split('.')[1] == 'csv']
            
            # Check if scores_data exists, else, add the column names 
            storing_df =  pd.read_csv('scores_data.csv') if os.path.isfile('scores_data.csv') else pd.DataFrame(columns=self.columns)
                
            # Loop across engines and runs and store the scores
            for engine in tqdm(engines):
                print(f'Fitting for engine: {engine}-------------------------------------------')
                path = f'data/{engine}.csv'
                full_df = pd.read_csv(path)
                full_df['participant'] = 0
                no_participants = full_df['participant'].max() + 1 
                # With LLMs actually we only have one participant so the for loop is useless, but this is to keep in mind that it could be generalized for multiple participants
                for participant in range(no_participants):
                    df_run = full_df[full_df['participant'] == participant].reset_index(drop=True)
                    storing_df = self.get_scores(df_run, storing_df, engine, participant)
                    storing_df.to_csv('scores_data.csv', index=False)

    def get_scores(self, participant_df, storing_df, engine, participant):
        """Get the scores for the Horizon task with a logistic regression's exploration effects.
        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
            engine (str): Name of the engine used
            run (int): Number of the run
        Returns:
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
        """
        conditions = ['equal', 'unequal']
        # conditions = ['unequal']

        for i, mode in enumerate(conditions):
            reward_differences = []
            horizons = []
            choices = []
            for run in participant_df.run.unique():
                df = participant_df[participant_df.run == run].reset_index(drop=True)
                horizon = 1 if len(df) == 10 else 0
                choices_B = df[df.trial < 4].choice.sum()
                if mode == 'equal':
                    reward_difference = df.mean1[0] - df.mean0[0]
                    choice = int(df[df.trial == 4].choice.iloc[0])
                    if choices_B == 2:
                        reward_differences.append(reward_difference)
                        choices.append(choice)
                        horizons.append(horizon)
                else:
                    if choices_B == 1: #case: x3 A
                        reward_difference = df.mean1[0] - df.mean0[0]
                        choice = int(df[df.trial == 4].choice.iloc[0])

                        reward_differences.append(reward_difference)
                        choices.append(choice)
                        horizons.append(horizon)

                    if choices_B == 3: #case: x3 B
                        reward_difference = df.mean0[0] - df.mean1[0]
                        choice = 1 - int(df[df.trial == 4].choice.iloc[0])
                        reward_differences.append(reward_difference)
                        choices.append(choice)
                        horizons.append(horizon)

            horizons = (np.array(horizons) - 0.5)*2 #Avoids interaction effect equating to 0 if horizon = 0
            try: 
                log_reg = sm.Logit(np.array(choices), np.stack((np.array(reward_differences), np.array(horizons), np.array(reward_differences) * np.array(horizons), np.ones(np.array(reward_differences).shape)), axis=-1)).fit()
                if not log_reg.mle_retvals['converged']:
                    raise Exception(f'{engine} did not converge for logreg, must be due to quasi separation and therefore unreliable effects so set to 0')
                if mode == 'equal':
                    random_exploration = - log_reg.params[2] #- because the random exploration would be the effect of choosing the other arm
                    random_exploration_ci = log_reg.bse[2] * 1.96
                else:
                    directed_exploration = log_reg.params[1]
                    directed_exploration_ci = log_reg.bse[1] * 1.96              
            except:
                #prefect seperation error
                print('Seperation error')
                if mode == 'equal':
                    random_exploration = 0
                    random_exploration_ci = np.nan
                else:
                    directed_exploration = 0
                    directed_exploration_ci = np.nan
            
        # Add the final scores to the csv file
        rewards = participant_df.reward.mean()
        rewards_ci = participant_df.reward.std() * 1.96 / np.sqrt(len(participant_df))
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == participant)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'performance_score1'] = rewards
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score1'] = directed_exploration
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score2'] = random_exploration
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score1_CI'] = directed_exploration_ci
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score2_CI'] = random_exploration_ci
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'performance_score1_CI'] = rewards_ci
        else:
            storing_df.loc[len(storing_df)] = [engine, participant, rewards, 'rewards mean', directed_exploration, 'Directed Exploration', random_exploration, 'Random Exploration', directed_exploration_ci, random_exploration_ci, rewards_ci]
        return storing_df

if __name__ == '__main__':
    StoringExplorationScores().get_all_scores()