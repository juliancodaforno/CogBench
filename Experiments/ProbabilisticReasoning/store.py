import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringPriorLhScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['behaviour_score2','behaviour_score2_name', 'behaviour_score3', 'behaviour_score3_name','behaviour_score4' , 'behaviour_score4_name'], help='Which scores to analyze')

    def get_all_scores(self):
        """
        Overides the baseclass function because regression across all runs makes more sense rather than for each run
        """
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
            storing_df = self.get_scores(full_df, storing_df, engine, run=0)
            storing_df.to_csv('scores_data.csv', index=False)
    
    def get_scores(self, df, storing_df, engine, run):
        """
        Get scores for Theory of Learning to Infer task across all runs.

        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores
            engine (str): Engine name
            run (int): Run number
        Returns:    
            storing_df (pd.DataFrame): Dataframe with the scores

        The function calculates the log odds of the prior, likelihood, and predicted probabilities. 
        It then performs a least squares regression to get the beta coefficients, which represent the weighting of lh and priors. 
        """ 

        if engine != 'human':
            # if red_observation == False then the ball is blue and therefore the lh is 1 - lh for calculating the posterior
           df.loc[df['red_observation'] == False, 'lh'] = df.loc[df['red_observation'] == False, 'lh'].apply(lambda x: 1 - x)

        # get the log odds actually
        prior = np.log(df['prior'] / (1-df['prior']))
        likelihood = np.log(df['lh'] / (1-df['lh'])) 
        #bound preds between 0.01 and 0.99
        df['left_pred'] = df['left_pred'].apply(lambda x: 0.00001 if x < 0.00001 else x)
        df['left_pred'] = df['left_pred'].apply(lambda x: 0.99999 if x > 0.99999 else x)
        post_prob= np.log(df['left_pred']/ (1-df['left_pred']))
        left_post = df['prior'] * df['lh'] / (df['prior'] * df['lh'] + (1-df['prior']) * (1-df['lh']))

        # regress to get the beta coefficients for prior and likelihood which represent the weighting of each
        X = np.column_stack((prior, likelihood))
        # Least squares regression
        lr_model=sm.OLS(post_prob, X)
        result=lr_model.fit()

        if engine == 'human': #Different for human because I stored it differently
            #if red_observation == False then the ball is blue and therefore the lh is 1 - lh for calculating the posterior
            df.loc[df['red_observation'] == False, 'lh'] = df.loc[df['red_observation'] == False, 'lh'].apply(lambda x: 1 - x)

        #Performance score 1: 1 - Difference to Bayes optimal prediction
        left_post = df['prior'] * df['lh'] / (df['prior'] * df['lh'] + (1-df['prior']) * (1-df['lh']))
        error = np.abs(left_post - df['left_pred'])
        print(f" Performance:{1 - error.mean()} +- {1.96 * np.std(error) / np.sqrt(len(error))}")

        prior_fit = result.params.iloc[0]
        lh_fit = result.params.iloc[1]

        if np.isnan(prior_fit):
            prior_fit = 0
        if np.isnan(lh_fit):
            lh_fit = 0
        if np.abs(lh_fit) <= 0.001 or np.isnan(result.params.iloc[0]):
            ratio = 0
        else:
            ratio = prior_fit / lh_fit

        if (lh_fit == 0) or (prior_fit == 0):
            #Avoids division by 0
            ratio_ci = 0
        else:
            ratio_ci = 1.96 * np.sqrt((result.bse.iloc[0] / prior_fit)**2 + (result.bse.iloc[1] / lh_fit)**2) * np.abs(prior_fit / lh_fit) # SE(X/Y) = sqrt[(SE(X)/X)^2 + (SE(Y)/Y)^2] * |X/Y|

        # print all metrics 
        print(f'Prior fit: {prior_fit} +/- {result.bse.iloc[0] * 1.96}')
        print(f'Likelihood fit: {lh_fit} +/- {result.bse.iloc[1] * 1.96}')
        print(f'Ratio: {ratio} +/- {ratio_ci}')
        
        # Add the final score to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = 1 - error.mean()
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1_CI'] = 1.96 * np.std(error) / np.sqrt(len(error))
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = prior_fit
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2'] = lh_fit
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] ==run), 'behaviour_score1_CI'] = result.bse.iloc[0] * 1.96
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2_CI'] = result.bse.iloc[1] * 1.96 
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score3'] = ratio
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score4'] = prior_fit - lh_fit
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score3_CI'] = ratio_ci
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score4'] = prior_fit - lh_fit
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score4_CI'] = 1.96 * np.sqrt(result.bse.iloc[0]**2 + result.bse.iloc[1]**2)
        else:
            storing_df.loc[len(storing_df)] = [engine, run, 1-  error.mean(), '1 - Difference to bayes optimal posterior prediction', 1.96 * np.std(error) / np.sqrt(len(error)),prior_fit, 'prior weighting',result.bse.iloc[0]*1.96, lh_fit, 'likelihood weighting', result.bse.iloc[1] * 1.96,
                                               ratio, 'prior / likelihood', ratio_ci,prior_fit - lh_fit, 'prior - likelihood', 1.96 * np.sqrt(result.bse.iloc[0]**2 + result.bse.iloc[1]**2)]
        return storing_df

if __name__ == '__main__':
    StoringPriorLhScores().get_all_scores()

