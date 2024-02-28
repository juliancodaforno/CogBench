"""
This script is used to compare the performance of the baseline vs step-back and CoT prompted models in the Probabilistic Reasoning  and TwoStepTask experiment.
"""
import numpy as np
import pandas as pd
from utils import comparison_barplot
import sys
import warnings
    

if __name__ == '__main__':
    #! Hyperparameters
    score_names = []
    # experiment = 'TwoStepTask'
    # score_names.append('behaviour_score1')
    experiment = 'ProbabilisticReasoning'
    score_names.append('performance_score1')
    base_agents = [  'gpt-4', 'text-bison@002', 'claude-2', 'claude-1', 'llama-2-70']
    including_agents = [agent + '_cot' for agent in base_agents] + [agent + '_sb' for agent in base_agents] + base_agents
    # including_agents = 'all' # if all agents are wanted


    #! Merge scores data with llm features
    df = pd.read_csv(f'../Experiments/{experiment}/scores_data.csv')
    llm_df = pd.read_csv(f'./data/llm_features.csv')
    increasing_param_feature = 'No of Parameters'
    # Add column names to the df
    df[increasing_param_feature] = np.nan
    df['Use of RLHF'] = np.nan
    df['Open Source'] = np.nan
    df['Chain of Thought'] = np.nan
    df['Step Back'] = np.nan
    df['Finetuned version of LLM'] = np.nan
    for idx , row in enumerate(df.iterrows()):
        engine = row[1]['engine']
        # Add new columns info
        try: 
            warnings.simplefilter(action='ignore', category=FutureWarning) # ignore Futurewarning for pandas
            df.loc[idx, increasing_param_feature] = llm_df.loc[llm_df['Engine'] == engine, increasing_param_feature].values[0]
            df.loc[idx, 'Use of RLHF'] = llm_df.loc[llm_df['Engine'] == engine, 'Use of RLHF'].values[0]
            df.loc[idx, 'Open Source'] = llm_df.loc[llm_df['Engine'] == engine, 'Open Source'].values[0]
            df.loc[idx, 'Chain of Thought'] = llm_df.loc[llm_df['Engine'] == engine, 'Chain of Thought'].values[0]
            df.loc[idx, 'Step Back'] = llm_df.loc[llm_df['Engine'] == engine, 'Step Back'].values[0]
            df.loc[idx, 'Finetuned version of LLM'] = llm_df.loc[llm_df['Engine'] == engine, 'Finetuned version of LLM'].values[0]
        except:
            print(sys.exc_info()[0])
            import ipdb; ipdb.set_trace()


    #! Prepare the data
    llm_groups = []
    llm_group_names = []
    df['No of Parameters'] = pd.to_numeric(df['No of Parameters'], errors='coerce')
    listofsorted_engines  = df.groupby('engine')['No of Parameters'].mean().sort_values(ascending=False).index.tolist()
    for engine in listofsorted_engines:
        if engine.endswith(('_cot')):
            sb = engine.replace('_cot', '_sb')
            baseline = engine.replace('_cot', '')
            #check all 3 engines in llm are in the df, if not skip this llm group
            if (baseline in df['engine'].unique().tolist()) and (sb in df['engine'].unique().tolist()):   
                # Only include llms wanted if not 'all'
                if including_agents == 'all' or (baseline in base_agents):
                    llm_group_names.append(baseline)
                    llm_groups.append([engine, sb, baseline])

    for score_name in score_names:
        #! Run the comparison barplot
        comparison_barplot(df, experiment, score_name, llm_groups, llm_group_names)