"""
This script is designed to perform multi-level regression analysis on performance scores of various LLMs engines across multiple experiments. 

To use this script, you need to store the information about the LLM engines you want to analyze and their features in a CSV file named 'llm_features.csv' in the 'data' directory. The CSV file should contain columns for the engine names and their features, with the first column named 'Engine'.

The script will read in the engine features from the CSV file and prepare the data by adding the engine features to the scores data. It will then filter the data to include only the desired engines and features, and standardize the features and scores.

The script will concatenate the standardized data for each experiment into a single DataFrame.

Finally, the script will perform regression on the performance scores and generate a bar plot of the results.

You can modify the list of engines to include in the analysis and the list of engines and features to exclude from the analysis as per your requirements.
"""


import pandas as pd
from utils import regression, prepare_dataframe, filter_dataframe, standardize_dataframe

# Run
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

experiments ={'RestlessBandit': ['performance_score1'],
                                'TwoStepTask': ['performance_score1'],
                                'HorizonTask': ['performance_score1'],
                                'BART': ['performance_score1'],
                                'ProbabilisticReasoning': ['performance_score1'],
                                # 'TemporalDiscounting': ['performance_score1'],
                                'InstrumentalLearning': ['performance_score1'],
                                }
perf_df = pd.DataFrame()
for experiment, scores_names in experiments.items():
    # Get the scores data for the experiment
    df = pd.read_csv(f"../Experiments/{experiment}/scores_data.csv")
    # Essentially adds the LLM features to the dataframe with the scores
    df, features = prepare_dataframe(df, llm_df, features_to_exclude)

    #only take the scores and LLMs we are interested in
    df = filter_dataframe(df, including_agents, excluding_agents, features, scores_names)
    #standardize the dataframe features and scores
    df_standardized, features = standardize_dataframe(df, features, scores_names)
    #concat the dataframes of each performance scores because regress on all scores at once
    perf_df = pd.concat([perf_df, df_standardized], ignore_index=True)

#rename performance_score1 to performance
perf_df = perf_df.rename(columns={'performance_score1': 'performance'})
#For performance, regress on all scores at once and store the bar plot
regression(perf_df, 'Concatenated Performance', 'performance', features, regtype='performance')


