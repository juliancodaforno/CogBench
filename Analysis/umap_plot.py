import pandas as pd
import numpy as np
from utils import umap_plot, merge_all_metrics_and_features

if __name__ == '__main__':
    # Hyperparameters
    interest = 'behaviour'
    # interest = 'performance'
    feature_of_interest = 'RLHF'
    excluding_agents = [ 'rational', 'meta-RL']
    # 'human', 'random']
    name_all = False #If True, all agents are named in the plot. If False, only the ones in the paper are named

    #! Llama 1 variants to exclude because not included in the paper since they were replaced by llama 2
    excluding_agents += ['llama_65', 'llama_30', 'llama_13', 'llama_7', 'vicuna_13', 'vicuna_7', 'hf_vicuna-7b-v1.3', 'hf_vicuna-13b-v1.3' , 'hf_vicuna-33b-v1.3', 'hf_koala-7B-HF', 'hf_koala-13B-HF']
    # excluding_agents.append('gemini-pro')
    #! Add all the _cot and _sb variants to the excluding agents
    df_llms = pd.read_csv('./data/llm_features.csv')
    for engine in df_llms['Engine'].unique().tolist():
        if engine.endswith('_cot') or engine.endswith('_sb'):
            excluding_agents.append(engine)

    #! The following includes the metrics of importance for behaviour and performance (listed in the paper) for each experiment. But this can be changed for exploring other metrics
    if interest == 'behaviour':
        experiments ={'RestlessBandit': ['behaviour_score3'],
                                            'TwoStepTask': ['behaviour_score1'],
                                            'HorizonTask': ['behaviour_score1', 'behaviour_score2'],
                                            'BART': ['behaviour_score1'],
                                            'ProbabilisticReasoning': ['behaviour_score1', 'behaviour_score2'],
                                            'TemporalDiscounting': ['performance_score1'],
                                            'InstrumentalLearning': ['behaviour_score1','behaviour_score2'],
                                            }
    elif interest == 'performance':
        experiments ={'RestlessBandit': ['performance_score1'],
                                            'TwoStepTask': ['performance_score1'],
                                            'HorizonTask': ['performance_score1'],
                                            'BART': ['performance_score1'],
                                            'ProbabilisticReasoning': ['performance_score1'],
                                            # 'TemporalDiscounting': ['performance_score1'], #This is more of a behaviour score
                                            'InstrumentalLearning': ['performance_score1'],
                                            }

    # Merge the metrics and the features
    metrics = merge_all_metrics_and_features(experiments, excluding_agents, df_llms)

    # Add  feature of interest 
    feature_of_interest_name = 'Use of RLHF'
    feature_of_interest = []
    for engine in metrics.keys():
        feature_of_interest.append(df_llms.loc[df_llms['Engine'] == engine][feature_of_interest_name].values[0])

    umap_plot(metrics, feature_of_interest, feature_of_interest_name, interest, name_all=name_all)



