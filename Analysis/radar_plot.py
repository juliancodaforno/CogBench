"""
This script is designed to perform radar plot analysis on behavior and performance scores of various LLMs engines across multiple experiments. 
Was not used in the paper but can be used to compare the behaviour and performance of the LLMs engines across multiple experiments.
"""
import pandas as pd
import numpy as np
from utils import radar

# Run
interest = 'behaviour'
# interest = 'performance'
agents_to_plot = ['gpt-4', 'llama-2-70-chat', 'claude-2']
excluding_agents = [ 'rational', 'meta-RL'] #The ones not used for squashing
#! Llama 1 variants to exclude
excluding_agents += ['llama_65', 'llama_30', 'llama_13', 'llama_7', 'vicuna_13', 'vicuna_7', 'hf_vicuna-7b-v1.3', 'hf_vicuna-13b-v1.3' , 'hf_vicuna-33b-v1.3', 'hf_koala-7B-HF', 'hf_koala-13B-HF']
#! Add all the _cot and _sb variants to the excluding agents
df_llms = pd.read_csv('./data/llm_features.csv')
for engine in df_llms['Engine'].unique().tolist():
    if engine.endswith('_cot') or engine.endswith('_sb'):
        excluding_agents.append(engine)

#! for now extras
if interest == 'behaviour':
    labels = ['Meta-cognition', 'Model-basedness', 'Directed exploration', 'Random explo', 'Risk', 'Prior weighting','Likelihood weighting', 'Temporal discounting', 'Learning rate', 'Optimism bias']
else:
    labels = ['Meta-cognition', 'Model-basedness', 'Exploration', 'Risk', 'Bayesian update', 'Learning rate']

# Define the experiments and the scores to use for each experiment. Here are the relevant scores listed in the paper, but one can explore other scores as well
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

# Add llm features to the df by checking which engine it is and adding the features columns
llm_df = pd.read_csv(f'./data/llm_features.csv')
#Including all agents
including_agents = llm_df['Engine'].unique().tolist()
#delete the agents we don't want
metrics = {engine:[] for engine in including_agents if engine not in excluding_agents}
for experiment, scores_names in experiments.items():
    # Get the scores data for the experiment
    df = pd.read_csv(f"../Experiments/{experiment}/scores_data.csv")

    for score_name in scores_names:
        for engine in metrics.keys():
            # Get the scores for the engine
            engine_scores = df[df['engine'] == engine][score_name]
            #! Hardcoded because do not have the scores for the human in the BART experiment so refered to the (Lejuez et al., 2002) paper for the scores
            if (experiment == 'BART') & (engine == 'human'):
                if score_name == 'behaviour_score1':
                    #Blue,yellow and orange ballons. I only had the earning + explosions, so this gives me best estimate of number of pumps:
                    men = 32.2*(1+9.1/30) + 7.6*(1+16.5/30) + 1.2*(1+22.2/30)
                    women = 26.4*(1+7.6/30) + 8*(1+15/30) + 1.4*(1+21.7/30)
                    human_pumps = (men+women)/6
                    metrics['human'].append(human_pumps)
                elif score_name == 'performance_score1':
                    metrics['human'].append((32.2+7.6+1.2+26.4+8+1.4)/6)
            else:
                metrics[engine].append(np.mean(engine_scores))
                if np.isnan(np.mean(engine_scores)):
                    print(f"Engine {engine} has NaN values for {score_name} in experiment {experiment}")
                    import ipdb; ipdb.set_trace()

radar(metrics, agents_to_plot, interest, labels)




