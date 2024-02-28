import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
import statsmodels.api as sm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringTemporalDiscountingScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['behaviour_score2','behaviour_score2_name','behaviour_score3','behaviour_score3_name','behaviour_score4', 'behaviour_score4_name'], help='List of columns to add to the csv file')
    
    def get_scores(self, df, storing_df, engine, run):
        """
        Get scores for temporal discounting task

        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores
            engine (str): Engine name
            run (int): Run number
        Returns:    
            storing_df (pd.DataFrame): Dataframe with the scores
        """
        #Create the columns if does not exist
        for column in self.parser.parse_args().columns:
            if column not in storing_df.columns:
                storing_df[column] = np.nan

        # Add the final score to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = df.score[0]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = df['present bias'][0]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2'] = df['subaddictivity'][0]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score3'] = df['delay-speedup asymmetry-1'][0]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score4'] = df['delay-length asymmetry-2'][0]
        else:
            storing_df.loc[len(storing_df)] = [engine, run, df.score[0], 'temporal discounting score', df['present bias'][0], 'present bias', df['subaddictivity'][0], 'subaddictivity', df['delay-speedup asymmetry-1'][0], 'delay-speedup asymmetry-1', df['delay-length asymmetry-2'][0], 'delay-length asymmetry-2']
        return storing_df
            
if __name__ == '__main__':
    StoringTemporalDiscountingScores().get_all_scores()