import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores


class StoringRiskScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['behaviour_score2','behaviour_score2_name'])
        self.parser.add_argument('--no_runs_counting_for_one_participant', nargs='+', default=320, help='Number of runs that count as one participant')


    def get_scores(self, df, storing_df, engine, participant):
        """Get the scores for the BART task.
        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
            engine (str): Name of the engine used
            run (int): Number of the run
        Returns:
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
        """
        # Get the scores
        adj_risk = np.mean(df[df['exploded'] == False]['reward'])
        rewards = np.mean(df['reward'])
        pumps = np.mean(df['pumps'])
        
        # Add the final scores to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == participant)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'performance_score1'] = rewards
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score1'] = pumps
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == participant), 'behaviour_score2'] = adj_risk
        else:
            storing_df.loc[len(storing_df)] = [engine, participant, rewards, 'Average rewards', pumps, 'Risk', adj_risk, 'Adj Risk']
        return storing_df
    
if __name__ == '__main__':
    StoringRiskScores().get_all_scores()