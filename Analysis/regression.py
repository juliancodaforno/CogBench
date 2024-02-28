"""
This script is designed to perform multi-levelregression analysis on behavior scores of various LLMs engines across multiple experiments. 

To use this script, you need to store the information about the LLM engines you want to analyze and their features in a CSV file named 'llm_features.csv' in the 'data' directory. The CSV file should contain columns for the engine names and their features, with the first column named 'Engine'.

The script will read in the engine features from the CSV file and prepare the data by adding the engine features to the scores data. It will then filter the data to include only the desired engines and features, and standardize the features and scores.

Finally, the script will perform regression on each behavior scores and generate a bar plot of the results.

You can modify the list of engines to include in the analysis and the list of engines and features to exclude from the analysis as per your requirements.
"""

import pandas as pd
import numpy as np
from utils import regression, prepare_dataframe, filter_dataframe, standardize_dataframe

# Run
experiments ={'RestlessBandit': ['behaviour_score3'],
                                    'TwoStepTask': ['behaviour_score1'],
                                    'HorizonTask': ['behaviour_score1', 'behaviour_score2'],
                                    'BART': ['behaviour_score1'],
                                    'ProbabilisticReasoning': ['behaviour_score1', 'behaviour_score2', 'behaviour_score3', 'behaviour_score4'],
                                    'TemporalDiscounting': ['performance_score1'],
                                    'InstrumentalLearning': ['behaviour_score1','behaviour_score2','behaviour_score3', 'behaviour_score4']}

llm_df = pd.read_csv(f'./data/llm_features.csv')
#Including all agents
including_agents = llm_df['Engine'].unique().tolist()
#Hyperparameters:
excluding_agents=['human', 'random', 'rational', 'meta-RL', 'gemini-pro']
excluding_agents += ['gpt-4_cot', 'gpt-4_sb', 'text-bison_cot', 'text-bison_sb', 'claude-1_cot', 'claude-1_sb', 'claude-2_cot', 'claude-2_sb', 'llama-2-70_cot', 'llama-2-70_sb', 'llama-2-70-chat_cot','llama-2-70-chat_sb', 'hf_Llama-2-70b-longlora-32k_cot', 'hf_Llama-2-70b-longlora-32k_sb', 'hf_Llama-2-70b-chat-longlora-32k_cot', 'hf_Llama-2-70b-chat-longlora-32k_sb', 'hf_LongAlpaca-70B_cot', 'hf_LongAlpaca-70B_sb']
# excluding features. the full list is : ['Use of RLHF', 'Open Source', 'conversational', 'code', 'longlora', 'longQA']
features_to_exclude = ['Engine'] #!Engine always has to be here!!!
# features_to_exclude += ['conversational', 'code', 'longlora', 'longQA']
features_to_exclude += ['Step Back','Chain of Thought']

for experiment, scores in experiments.items():
    # Get the scores data for the experiment
    df = pd.read_csv(f"../Experiments/{experiment}/scores_data.csv")
    df, features = prepare_dataframe(df, llm_df, features_to_exclude)

    #Get score names for barplot
    score_names = [df[f'{score}_name'].values[0].replace('/', 'divided by') for score in scores]
    #only take the scores and LLMs we are interested in
    df = filter_dataframe(df, including_agents, excluding_agents, features, scores)
    #standardize the dataframe features and scores
    df, features = standardize_dataframe(df, features, scores)

    # Run regression for each score 
    for score, score_name in zip(scores, score_names):
        regression(df, experiment, score, features, score_name=score_name)




