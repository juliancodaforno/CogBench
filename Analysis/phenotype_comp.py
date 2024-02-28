import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import merge_all_metrics_and_features

def normalize_metrics(df, metrics):
    for metric in metrics:
        df[metric] = (df[metric] - df.loc[df['Agent'] == 'random', metric].values)
        df[metric] /= df.loc[df['Agent'] == 'human', metric].values

    return df

def exclude_agents():
    '''
    Filter out the engines in llm_features.csv for which we don't have metrics.
    '''
    excluding_agents = [ 'rational', 'meta-RL'] #These engines were exploratory for specific experiments and therefore do not have metrics for all experiments.
    #! Llama 1 variants to exclude because not included in the paper since they were replaced by llama 2
    excluding_agents += ['llama_65', 'llama_30', 'llama_13', 'llama_7', 'vicuna_13', 'vicuna_7', 'hf_vicuna-7b-v1.3', 'hf_vicuna-13b-v1.3' , 'hf_vicuna-33b-v1.3', 'hf_koala-7B-HF', 'hf_koala-13B-HF']
    #! Add all the _cot and _sb variants to the excluding agents
    df_llms = pd.read_csv('./data/llm_features.csv')
    for engine in df_llms['Engine'].unique().tolist():
        if engine.endswith('_cot') or engine.endswith('_sb'):
            excluding_agents.append(engine)
    return excluding_agents

def prepare_data_for_plotting(df, metrics):
    agents = df.Agent.unique()
    rows = []
    for metric in metrics:
        for agent in agents:
            value = df[df['Agent'] == agent][metric].item()
            rows.append([value, agent, metric])

    return pd.DataFrame(rows, columns=['Value', 'Model', 'Task'])

def filter_models(dp, models):
    dp = dp[~dp['Model'].isin(['human', 'random']) & dp['Model'].isin(models)]
    return dp

def plot_data(dp, filename,metrics_names, behav=False, store_id='0'):
    models = dp['Model'].unique()
    n_models = len(models)
    # fig, axs = plt.subplots(1, n_models, figsize=(1.5 * n_models, 3.5))
    fig, axs = plt.subplots(1, n_models, figsize=(10.5, 3.5))
    # fig, axs = plt.subplots(1, n_models, figsize=(7, 7))

    if behav:
        # metrics_perf = ['Bayesian update', 'Exploration', 'Meta-cognition',  'Learning rate', 'Model-basedness','Risk']
        colors = ['tab:green', 'tab:green', 'tab:orange', 'tab:orange', 'tab:blue', 'tab:red', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple']  # Add more colors if needed
    else:
        colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:brown', 'tab:purple']  # Add more colors if needed

    models_names = ["GPT-4", "text-davinci-003", "Claude-2", "Claude-1", "text-bison", "LLaMA-2-70", "LLaMA-2-70-chat"]
    for i, (ax, model) in enumerate(zip(axs, models)):
        dp_model = dp[dp['Model'] == model]
        sns.barplot(data=dp_model, x='Value', y='Task', ax=ax, alpha=0.6, hue='Task', palette=colors, dodge=False,)
        if store_id == '0':
            ax.set_title(models_names[i], fontsize=10)
        else:
            if len(model) > 20:
                model = f'{model[:len(model)//2]}\n{model[len(model)//2:]}'
            ax.set_title(model, fontsize=10)
        ax.axvline(x=1, color='black', linestyle='dotted')
        ax.axvline(x=0, color='black', linestyle='dotted')
        # ax.legend().remove()
        if behav:
            ax.set_xlim(-1, 5)
        else:
            ax.set_xlim(-1, 3)
        #Also get rid of "Model" label
        ax.set_ylabel('')
        if i != 0:  # if not the leftmost subplot
            ax.set_yticklabels([])  # remove the y-tick labels
            ax.set_xlabel('')
        else:
            ax.set_yticks(range(len(metrics_names)))
            ax.set_yticklabels(metrics_names)
            ax.set_xlabel('                                                                                                                                                                                                       Values (Normalized: Random=0, Human=1)')    

    plt.tight_layout()
    plt.savefig(f'./plots/phenotypes/{store_id}{filename}')

def run():
    """
    Main function to compare models based on behavior or performance.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare models and optionally print scores.')
    parser.add_argument('--models', nargs='+', default=["gpt-4", "text-davinci-003", "claude-2", "claude-1", "text-bison@002", "llama-2-70", "llama-2-70-chat"], help='List of models to compare. You need at least two models to compare and we recommend at most 7!')
    parser.add_argument('--print_scores', action='store_true', help='If set, print the scores for each model.')
    parser.add_argument('--print_scores_for', nargs='+', default=["gpt-4", "human", "random"], help='List of models for which to print scores.')
    parser.add_argument('--store_id', type=str, default='', help='If set, store the plot with this unique identifier.')
    parser.add_argument('--interest', choices=['behaviour', 'performance'], default='behaviour', help='Specify whether you are interested in "behavior" or "performance" plots.')
    
    args = parser.parse_args()

    # Extract arguments
    models = args.models
    print(models)
    interest = args.interest
    store_id = args.store_id

    # Define metrics and experiments based on the interest. Here the default are the ones in the paper.
    if interest == 'behaviour':
        experiments = {
            'ProbabilisticReasoning': ['behaviour_score1', 'behaviour_score2'],
            'HorizonTask': ['behaviour_score1', 'behaviour_score2'],
            'RestlessBandit': ['behaviour_score3'],
            'InstrumentalLearning': ['behaviour_score1','behaviour_score2'],
            'TwoStepTask': ['behaviour_score1'],
            'TemporalDiscounting': ['performance_score1'],
            'BART': ['behaviour_score1'],
        }
        metrics_names =  ['Prior weighting', 'Likelihood weighting','Directed exploration', 'Random exploration', 'Meta-cognition', 'Learning rate', 'Optimism bias', 'Model-basedness',  'Temporal discounting','Risk taking']
        output='behaviour.pdf'
    elif interest == 'performance':
        experiments = {
            'ProbabilisticReasoning': ['performance_score1'],
            'HorizonTask': ['performance_score1'],
            'RestlessBandit': ['performance_score1'],
            'InstrumentalLearning': ['performance_score1'],
            'TwoStepTask': ['performance_score1'],
            # 'TemporalDiscounting': ['performance_score1'], # This is more of a behavioral task
            'BART': ['performance_score1'],
        }
        metrics_names = ['Probabilistic Reasoning', 'Horizon Task', 'Restless Bandit', 'Instrumental Learning', 'Two Step Task', 'BART']
        output='performance.pdf'

    # Merge all metrics and features
    metrics = merge_all_metrics_and_features(experiments, exclude_agents(), pd.read_csv('./data/llm_features.csv'))

    # Create a dataframe with the metrics
    df = pd.DataFrame(metrics).T
    df.columns = metrics_names
    df = df.reset_index().rename(columns={'index': 'Agent'})

    # Print scores for specified models
    if args.print_scores:
        print_scores_for = args.print_scores_for
        print("Scores before normalization:")
        pd.set_option('display.max_columns', 10) # Display all columns
        print(df[df['Agent'].isin(print_scores_for)].set_index('Agent'))

    # Uncomment the following line if you want to store the DataFrame as a CSV file
    # df.to_csv(f'./data/{interest}.csv'); df = pd.read_csv(f'./data/{interest}.csv')

    # Normalize metrics and prepare data for plotting
    df = normalize_metrics(df, metrics_names)
    dp = prepare_data_for_plotting(df, metrics_names)
    dp = filter_models(dp, models)

    # Plot data
    plot_data(dp, output,  metrics_names, interest=='behaviour', store_id)


if __name__ == "__main__":
    run()