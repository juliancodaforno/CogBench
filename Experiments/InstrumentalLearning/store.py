import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from scipy.optimize import differential_evolution
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringOptimismBias_and_LR(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['behaviour_score2','behaviour_score2_name'])

    def get_scores(self, df, storing_df, engine, run):
        """
        This function calculates and stores the performance and behavior scores for the experiment.

        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment.
            storing_df (pd.DataFrame): Dataframe where the scores will be stored.
            engine (str): The name of the engine used in the experiment.
            run (int): The run number of the experiment.

        Returns:    
            storing_df (pd.DataFrame): Dataframe with the updated scores.

        The function calculates four behavior scores and one performance score (more can be added if needed):
        - Behavior score 1: Learning rate, obtained by fitting parameters to the data without considering plus/minus.
        - Behavior score 2: Optimism bias, obtained by fitting parameters to the data considering plus/minus.
        - Behavior score 3: Difference with optimal learning rate of a Rescorla-Wagner agent.
        - Behavior score 4: Difference with optimal optimism bias of a Rescorla-Wagner agent with plus/minus.
        - Performance score 1: Mean accuracy, calculated as the mean of the 'reward' column in the dataframe.

        The scores are then added to the storing dataframe. If the engine and run already exist in the storing dataframe, the scores are updated. Otherwise, a new row is added with the scores.

        The function returns the updated storing dataframe.
        """

        # Behaviour score 1: Learning rate
        params_rw, _ = self.fit_parameters(df, plus_minus=False)

        # Behaviour score 2: Optimism bias
        params_rwpm, _ = self.fit_parameters(df, plus_minus=True)

        #Behaviour score 3: Difference with optimal rw agent's LR
        lr_optimal = 0.069 #inverse temperature = 316

        # Behaviour score 4: Difference with optimal rwpm agent's LR
        optimism_bias_optimal = 0.172#inverse temperature = 17

        # Performance score 1: Mean accuracy
        mean_reward = df['reward'].mean()

        # Add the final score to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = mean_reward
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = params_rw[0]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2'] = params_rwpm[0] - params_rwpm[1]
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score3'] = np.abs(params_rw[0] - lr_optimal)
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score4'] = np.abs(params_rwpm[0] - params_rwpm[1] - optimism_bias_optimal)
        else:
            storing_df.loc[len(storing_df)] = [engine, run, mean_reward, 'mean_reward', params_rw[0], 'learning_rate', params_rwpm[0] - params_rwpm[1], 'optimism_bias', np.abs(params_rw[0] - lr_optimal), 'difference to optimal LR', np.abs(params_rwpm[0] - params_rwpm[1] - optimism_bias_optimal), 'difference to optimal optimism bias']
        return storing_df
            
    def rescorla_wagner(self, params, choices, outcomes, plus_minus):
        '''
        Rescorla wagner model. Given a set of choices and outcomes, it returns the log likelihood of the model.
        Args:
            params (list): list of parameters to fit
            choices (list): list of choices
            outcomes (list): list of outcomes
            plus_minus (bool): whether to fit the plus minus model or not
        Returns:
            log_likelihood (float): log likelihood of the model
        '''
        if plus_minus:
            alpha_plus, alpha_minus, inverse_temperature = params
        else:
            learning_rate, inverse_temperature = params
        
        # Initialize values
        values = np.ones(2)/2 # Because rewards are either 0 or 1 and we want to start with an average value.
        
        # Log-likelihood
        log_likelihood = 0
        for choice, outcome in zip(choices, outcomes):

            # Softmax computation with normalization
            logits = torch.tensor([inverse_temperature * values[0], inverse_temperature * values[1]])
            logits = logits - torch.max(logits)  # Subtract the maximum value for numerical stability
            probabilities = F.softmax(logits, dim=0)
            likelihood = probabilities[choice]
            epsilon = 1e-5
            likelihood = np.clip(likelihood, epsilon, 1-epsilon)
            log_likelihood += torch.log(likelihood)

            # Update values
            prediction_error = outcome - values[choice]
            if plus_minus:
                if prediction_error > 0:
                    values[choice] +=  alpha_plus * prediction_error
                else:
                    values[choice] +=  alpha_minus * prediction_error
            else: 
                values[choice] += learning_rate * prediction_error

        return log_likelihood


    def nll(self, params, df, plus_minus):
        '''
        Get the negative log likelihood of the rescorla wagner model.
        Args:
            params (list): list of parameters to fit
            df (pd.DataFrame): dataframe with the data
            plus_minus (bool): whether to fit the plus minus model or not
        Returns:
            nll (float): negative log likelihood of the model
        '''
        log_likelihood = 0
        for task in df.task.unique():
            df_task = df[df['task'] == task]
            choices = df_task['choice']
            outcomes = df_task['reward']
            log_likelihood += self.rescorla_wagner(params, choices, outcomes, plus_minus)
        return -log_likelihood

    def fit_parameters(self, df, plus_minus=False):
        """
        Fit the parameters of the rescorla wagner model.
        Args:
            df (pd.DataFrame): dataframe with the data
            plus_minus (bool): whether to fit the plus minus model or not
        Returns:
            best_params (list): list of best parameters
            best_nll (float): negative log likelihood of the model
        """
        # Initial parameter values
        bounds = [(0, 1), (0, 1e5)] if not plus_minus else [(0, 1), (0, 1), (0, 1e5)]
        result = differential_evolution(self.nll, bounds=bounds, args=(df, plus_minus), maxiter=100)
        best_nll = result.fun
        best_params = result.x
        return best_params, best_nll

    def optimal_agent(self, no_sims=1000):
        """
        This function performs a grid search to find the optimal parameters for both the standard Rescorla-Wagner model and the plus-minus variant.

        Args:
            no_sims (int): The number of simulations to run. Default is 1000.

        The function runs the grid search for the standard Rescorla-Wagner model and the +- model. 
        For each combination of learning rate and inverse temperature, 
        it simulates an agent making decisions in the environment and updates the agent's values based on the observed rewards. 
        The mean reward for each combination of parameters is then saved to a CSV file.
        """
        import csv
        self.parser.add_argument('--script_number', type=int, default=1, help='Script ran and therefore helps with name to save')
        script = self.parser.parse_args().script_number
        # Grid search for the optimal parameters
        learning_rates = np.linspace(0, 1, 30)
        inverse_temperatures = np.logspace(0, 5, num=5)

        #Create dataframe to store. Therefore columns are lr, inverse_temperature, mean_reward, run and lr2 for rwpm
        path_rw = f'./grid_search_csvs/mean_rewards_rw{script}.csv'
        path_rwpm = f'./grid_search_csvs/mean_rewards_rwpm{script}.csv'
        if os.path.isfile(path_rw):
            df_rw = pd.read_csv(path_rw)
            start_sim_rw = df_rw.run.max() + 1
        else:
            pd.DataFrame(columns=['run','lr', 'inverse_temperature', 'mean_reward']).to_csv(path_rw, index=False)
            start_sim_rw = 0
        if os.path.isfile(path_rwpm):
            df_rwpm = pd.read_csv(path_rwpm)
            start_sim_rwpm = df_rwpm.run.max() + 1
        else:
            pd.DataFrame(columns=['run','lr1', 'lr2', 'inverse_temperature', 'mean_reward']).to_csv(path_rwpm, index=False)
            start_sim_rwpm = 0

        # Standard rescorla wagner
        for simulation in range(start_sim_rw, no_sims):
            for i, lr in tqdm(enumerate(learning_rates)):
                for j, inverse_temperature in enumerate(inverse_temperatures):
                    import warnings
                    # Suppress warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        env = gym.make('palminteri2017-v0')
                        env.reset()
                    num_trials = env.max_steps
                    optimal_agent_values = torch.ones((4, 2))/2 
                    context = int(env.contexts[0, 0].numpy()) -1
                    for t in range(num_trials-1):
                        # Get the optimal action
                        logits = torch.tensor([inverse_temperature * optimal_agent_values[context, 0], inverse_temperature * optimal_agent_values[context, 1]])
                        logits = logits - torch.max(logits)  # Subtract the maximum value for numerical stability
                        probabilities = F.softmax(logits, dim=0)
                        probabilities[1] = 1.0 - probabilities[0]
                        choice = np.random.choice(2, p=probabilities.numpy())
                        observation, reward, _, _ =  env.step(choice) 

                        # Update values
                        optimal_agent_values[context, choice] += lr * (reward - optimal_agent_values[context, choice])
                        context = int(observation[0, 3]) -1

                    #Append to the csv file
                    new_row = [simulation, lr, inverse_temperature, reward]
                    with open(path_rw, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(new_row)

        # Plus minus rescorla wagner
        for simulation in range(start_sim_rwpm, no_sims):
            for i, lr1 in tqdm(enumerate(learning_rates)):
                for j, lr2 in enumerate(learning_rates):
                    for k, inverse_temperature in enumerate(inverse_temperatures):
                        env = gym.make('palminteri2017-v0')
                        env.reset()
                        num_trials = env.max_steps
                        optimal_agent_values = torch.ones((4, 2))/2 
                        context = int(env.contexts[0, 0]) - 1
                        for t in range(num_trials - 1):
                            # Get the optimal action
                            logits = torch.tensor([inverse_temperature * optimal_agent_values[context, 0], inverse_temperature * optimal_agent_values[context, 1]])
                            logits = logits - torch.max(logits)
                            probabilities = F.softmax(logits, dim=0)
                            probabilities[1] = 1.0 - probabilities[0]
                            choice = np.random.choice(2, p=probabilities.numpy())
                            observation, reward, _, _ =  env.step(choice)

                            # Update values
                            prediction_error = reward - optimal_agent_values[context, choice]
                            if prediction_error > 0:
                                optimal_agent_values[context, choice] += lr1 * prediction_error
                            else:
                                optimal_agent_values[context, choice] += lr2 * prediction_error
                            context = int(observation[0, 3]) - 1

                        #Append to the csv file
                        new_row = [simulation, lr1, lr2, inverse_temperature, reward]
                        with open(path_rwpm, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(new_row)

if __name__ == '__main__':
    StoringOptimismBias_and_LR().get_all_scores()