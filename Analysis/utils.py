import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
import os
import warnings

def p_to_stars(p):
    '''
      Function to convert p-values to significance stars
    '''
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""


def get_means_and_cis(df, score_name):
    '''
    Function to get the mean and confidence interval of a score from a dataframe
    Args:
        df: dataframe containing the score and maybe the confidence interval
        score_name: name of the score
    Returns:
        mean: mean of the score
        ci: confidence interval of the score
    '''
    mean = df[f'{score_name}'].mean() 
    std = df[f'{score_name}'].std()
    if f'{score_name}_CI' in df.columns:
        ci = df[f'{score_name}_CI'].values[0]
    else:
        ci = 1.96 * std / np.sqrt(len(df))
    ci = np.nan_to_num(ci)
    return mean, ci


def aggregated_results(dfs, score_name, experiment):
    aggregated_means = []
    weights = []
    weighted_mean_diffs = []
    cis_diffs = []

    for i, df in enumerate(dfs):
        type_of_cdt = 'use_cot' if i == 0 else 'use_sb'
        for model in df[df[type_of_cdt] == 0].engine.unique().tolist():
            dfbase = df[df['engine'] == model]
            dfcdt = df[df['engine'] == model + '_cot'] if i == 0 else df[df['engine'] == model + '_sb']
            meanbase = dfbase['score'].mean()
            meancdt = dfcdt['score'].mean()
            cibase = dfbase[f'{score_name}_CI'].values[0]
            cicdt = dfcdt[f'{score_name}_CI'].values[0]
            mean_diff = meancdt - meanbase
            ci_diff = np.sqrt(cibase**2 + cicdt**2)
            weight = 1 / ci_diff**2
            aggregated_means.append(mean_diff * weight)
            weights.append(weight)

        # Calculate the weighted mean difference
        weighted_mean_diff = np.sum(aggregated_means) / np.sum(weights)
        weighted_mean_diffs.append(weighted_mean_diff)

        # Calculate the CI for the weighted mean difference
        se = np.sqrt(1 / np.sum(weights))
        ci = 1.96 * se/2 #divide by 2 for radius of CI
        cis_diffs.append(ci)
        print(f'Aggregated mean difference for {type_of_cdt}: {weighted_mean_diff} +- {ci}')
    return weighted_mean_diffs, cis_diffs

def regression(df, experiment, score, features, score_name='n/a', regtype='behavior'):
    '''
    Performs a regression analysis on a given score using specified features and stores the results in a bar plot.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be analyzed.
    experiment (str): The name of the experiment being analyzed.
    score (str): The name of the column in df that contains the score to be analyzed.
    features (list of str): The names of the columns in df that contain the features to be used in the regression analysis.
    score_name (str): The name of the score to be used in saving the plot. If 'n/a', the name of the score is used.
    regtype (str): The type of regression to be performed. Can be 'behavior' or 'performance'.

    A mixed linear model is fitted using the standardized features and score, with the 'type_of_engine' column used as the groups.
    The function then extracts the coefficients, confidence intervals, and p-values from the fitted model.
    A bar plot of the coefficients is then created, with error bars representing the confidence intervals.
    The function finally saves the plot to the './plots/regressions/{experiment}' directory.
    '''
    plt.close()
    if score_name == 'n/a':
        score_name = score
    df_standardized = df.copy()

    # Set the style & size of the plots
    plt.rcParams["figure.figsize"] = (3.3,1.6)

    # The goal is to treat the 'Finetuned version of LLM' as a random effect in the mixed linear model.
    # For engines that are finetuned versions of other engines, we want to group them together under the original engine they were finetuned from.
    # For example, 'llama-2-chat-70' and 'llama-2-chat-70-finetuned' are both finetuned versions of 'llama-2-70'.
    # Therefore, we want to group them together under 'llama-2-70' as a random effect.
    # It's important to note that 'llama-2-70' itself should also be grouped under 'llama-2-70' as a random effect.

    # We start by getting a list of unique engine names, excluding 'No'.
    typesofengine = [engine for engine in df.engine.unique() if engine != 'No']

    # We create a new column 'type_of_engine' in the standardized DataFrame and initialize it with NaN values.
    df_standardized['type_of_engine']  = np.nan

    # For each row in the DataFrame:
    # If the engine is a finetuned version of another engine (i.e., 'Finetuned version of LLM' is not 'No'), 
    # we set 'type_of_engine' to the name of the engine it is a finetuned version of.
    # If 'Finetuned version of LLM' is not in the list of unique engine names, we set 'type_of_engine' to the value of 'Finetuned version of LLM'.
    for row in df.iterrows():
        if df['Finetuned version of LLM'][row[0]] != 'No':
            df_standardized['type_of_engine'][row[0]] = df.engine[row[0]] 
        if df['Finetuned version of LLM'][row[0]]  not in typesofengine:
            df_standardized['type_of_engine'][row[0]] = df['Finetuned version of LLM'][row[0]]

    # We replace spaces with underscores in the column names of the standardized DataFrame and in the features list.
    df_standardized.columns = df_standardized.columns.str.replace(' ', '_')
    features = [feature.replace(' ', '_') for feature in features]

    # We fit a mixed linear model to the standardized data, with 'score' as the dependent variable, 
    # the elements in the 'features' list as the independent variables, and 'type_of_engine' as the grouping variable.
    md = smf.mixedlm(f"{score} ~ { ' + '.join(features) }", df_standardized, groups=df_standardized['type_of_engine']) 
    mdf = md.fit(maxiter=10000)

    if not mdf.converged:
        print(f'Experiment {experiment} and score {score_name} did not converge')
        # import ipdb; ipdb.set_trace()

    # Extract the coefficients from the fitted model
    coefficients = mdf.params

    #Get rid of the intercept and group variance for the barplot 
    coefficients = coefficients[1:-1]
    cis = mdf.bse[1:-1] * 1.96
    # Extract the p-values from the fitted model
    pvalues = mdf.pvalues[1:-1]

    #Make coefficients names nicer
    coefficients.index = [feature.replace('_', ' ').title() for feature in coefficients.index]

    # Create a dataframe with feature names and their corresponding coefficients
    coeff_df = pd.DataFrame({'Feature': coefficients.index, 'Coefficient': coefficients.values})
    
    # Filter out 'longlora' and 'longqa' because too much variance for these two features effects
    coeff_df = coeff_df[~coeff_df['Feature'].isin(['Longlora', 'Longqa'])]
    cis = cis.drop(['longlora', 'longQA'])
    pvalues = pvalues.drop(['longlora', 'longQA'])

    # colors assignment for features of interest for paper due to hypothesized effects
    if experiment == 'RestlessBandit':
        features_of_interest = ['Use Of Rlhf']
    elif experiment == 'TwoStepTask':
        features_of_interest = ['No Of Parameters', 'Code', 'Size Of Dataset']
    elif experiment == 'BART':
        features_of_interest = ['Open Source']
    else:
        features_of_interest = ['No Of Parameters', 'Code', 'Size Of Dataset', 'Open Source', 'Use Of rlhf']
    # Plot a bar plot using seaborn
    barplot = sns.barplot(x='Coefficient', y='Feature', data=coeff_df, xerr=cis, color='black', alpha=0.6)
    
    # Change the color of the bars of interest 
    for i, feature in enumerate(coeff_df['Feature']):
        if feature in features_of_interest:
            barplot.patches[i].set_color('C03')

    # Add significance stars to the barplot
    for i, p in enumerate(pvalues):
            stars = p_to_stars(p)
            # Add the x position of the barplot
            x_pos = coeff_df['Coefficient'][i] + cis[i] if coeff_df['Coefficient'][i] > 0 else coeff_df['Coefficient'][i] - cis[i] 
            # Add the text annotation to the barplot
            plt.text(x_pos, i, stars, ha='center', va='bottom')

    # plt.title('Feature Coefficients onto' + f' {score_name}')
    experiments_with_yticks = ['RestlessBandit', 'Concatenated Performance'] 
    plt.xlabel('Regression coefficients', fontsize=10) 
    # plt.ylabel('LLMs features', fontsize=10)
    plt.ylabel('')
    plt.yticks(fontsize=7)
    #Hide yticks if not in experiments_with_yticks
    if experiment not in experiments_with_yticks:
        plt.yticks([])

    sns.despine()
    plt.xticks(fontsize=7)
    plt.tight_layout(pad=0.05)

    if regtype == 'behavior':
        os.makedirs(f'./plots/regressions/{experiment}', exist_ok=True)
        plt.savefig(f'./plots/regressions/{experiment}/barplot_{score_name}.pdf')
    else:
        plt.savefig(f'./plots/regressions/performance.pdf')


def prepare_dataframe(df, llm_df, features_to_exclude):
    """
    Prepares the DataFrame by adding new feature columns and filling them with appropriate values.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be analyzed.
    llm_df (pandas.DataFrame): The DataFrame containing the features of the language learning models.
    features_to_exclude (list of str): The names of the features to be excluded from the analysis.

    Returns:
    df (pandas.DataFrame): The prepared DataFrame.
    features (list of str): The names of the features used in the analysis.
    """
    #TODO: Got rid of warnings because verbose but just syntax problem
    warnings.filterwarnings("ignore")

    features = []
    for feature in llm_df.columns:
        if feature != 'Engine':
            df[feature] = np.nan
    for idx , row in df.iterrows():
        engine = row['engine']
        for feature in llm_df.columns:
            if feature not in features_to_exclude:
                if feature not in features:
                    features.append(feature)
                try: 
                    df.loc[idx, feature] = llm_df.loc[llm_df['Engine'] == engine, feature].values[0]
                    if feature in ['Open Source', 'Use of RLHF','conversational','code','longlora','longQA']:
                        if df.loc[idx, feature] == 'Yes':
                            df.loc[idx, feature] = 1
                        elif df.loc[idx, feature] == 'RLAIF':
                            df.loc[idx, feature] = 0.5
                        else:
                            df.loc[idx, feature] = 0
                except:
                    import ipdb; ipdb.set_trace()
    return df, features


def filter_dataframe(df, including_agents, excluding_agents, features, scores):
    """
    Filters the DataFrame based on the including and excluding agents, and the specified features and scores.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be analyzed.
    including_agents (list of str): The names of the agents to be included in the analysis.
    excluding_agents (list of str): The names of the agents to be excluded from the analysis.
    features (list of str): The names of the features to be used in the analysis.
    scores (str or list of str): The name(s) of the score(s) to be included in the analysis.

    Returns:
    df (pandas.DataFrame): The filtered DataFrame.
    """
    # Convert scores to a list if it's not already a list
    if not isinstance(scores, list):
        scores = [scores]
    # Filter the DataFrame to only include the specified scores, features, and 'engine'
    df = df[scores + features + ['engine']]
    # Only keep the engines that are included
    df = df[df['engine'].isin(including_agents)]
    # Exclude agents 
    for agent in excluding_agents:
        df = df[df['engine'] != agent]
    # Drop rows with nans
    df = df.dropna()
    return df


def standardize_dataframe(df, features, score):
    """
    Standardizes the features and the score in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be analyzed.
    features (list of str): The names of the features to be standardized.
    score (str): The name of the score to be standardized.

    Returns:
    df_standardized (pandas.DataFrame): The DataFrame with standardized features and score.
    features (list of str): The names of the standardized features.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # First get rid of feature finetuned version of LLM for standardization
    features = [feature for feature in features if feature != 'Finetuned version of LLM']
    # Fit and transform the features from the dataframe
    standardized_features = scaler.fit_transform(df[features])
    #Also standardize the score
    standardized_score = scaler.fit_transform(df[score])
    df[score] = standardized_score
    # Create a new dataframe with standardized features and the score
    df_standardized = df.copy()
    df_standardized[features] = standardized_features

    return df_standardized, features

def comparison_barplot(df,  experiment, score_name, engines, group_labels):
    """
    Create a barplot with confidence intervals for each group in the dataframe.
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to plot.
    experiment : str
        Name of the experiment.
    score_name : str
        Name of the column containing the score to plot.
    engines : list of lists
        List of lists containing the engines to plot. Each list will be a group.
    group_labels : list
        List containing the labels for each group.
    Returns
    -------
    None.
    """

    #Drop all engines that not in group_labels
    for engine in df['engine'].unique().tolist():
        if not engine.rstrip('_cot').rstrip('_sb') in group_labels:
            df = df[df['engine'] != engine]

    # Store means and confidence intervals to plot the min and max values
    means = []
    cis = []
    # Create a list to store the legend handles
    legend_handles = []

    plt.rcParams["figure.figsize"] = (3.3, 2)
    plt.rcParams['font.size'] = 9

    #? Storing data for regression of use of cot/sb on score
    dfcot = df.copy(); dfcot = dfcot.rename(columns={f'{score_name}': 'score'})
    dfsb = df.copy(); dfsb = dfsb.rename(columns={f'{score_name}': 'score'})
    #delete rows of other cdt
    dfcot = dfcot[dfcot['engine'].apply(lambda x: '_sb' not in x)]
    dfsb = dfsb[dfsb['engine'].apply(lambda x: '_cot' not in x)]
    #add column for use of cot/sb
    dfcot['use_cot'] = dfcot['engine'].apply(lambda x: 1 if '_cot' in x else 0)
    dfsb['use_sb'] = dfsb['engine'].apply(lambda x: 1 if '_sb' in x else 0)
    #Call aggregated results function to print results as well
    agg_means, agg_cis = aggregated_results([dfcot,dfsb], score_name, experiment)
    #? 
    #Loop across groups and plot
    for i, llm in enumerate(engines):
        mean_base, ci_base = get_means_and_cis(df[df['engine'] == llm[-1]], score_name)
        for idx, engine in enumerate(llm[:-1]):
            width=0.9
            mean, ci =  get_means_and_cis(df[df['engine'] == engine], score_name)
            mean = mean - mean_base
            ci = np.sqrt(ci**2 + ci_base**2)

            # Adding texture to the bars
            if '_cot' in engine:
                grey_int = 0.3
            elif '_sb' in engine:
                grey_int =0.7
            plt.bar(engine, mean, yerr=ci/2, width=width, color=f'C0{i}', alpha=grey_int)

            # Store means and confidence intervals to plot the min and max values
            means.append(mean)
            cis.append(ci)

        # Create a Patch object for each group and add it to the legend_handles list
        legend_handles.append(mpatches.Patch(color=f'C0{i}', label=group_labels[i], alpha=0.5))

    hatches = [mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='black', label=f'with CoT'),
            mpatches.Patch(facecolor='gray', alpha=0.7, edgecolor='black', label=f'with SB')]
    legend_handles = legend_handles + hatches
    # Set the y-axis label
    try:
        df_engine = df[df['engine'] == engine] # any engine will do
        plt.ylabel(f'{df_engine[f"{score_name}_name"].values[0]}')
    except:
        import ipdb; ipdb.set_trace()

    #! Mask some legends for some experiments - for paper!!!
    if (experiment == 'ProbabilisticReasoning') & (score_name == 'performance_score1'):
        plt.ylabel(f'Posterior accuracy \n($\Delta$ to no prompting)')
        legend_handles = legend_handles[:4]
        legend_handles[1] = mpatches.Patch(color=f'C01', label=f'text-bison', alpha=0.5)
        legend_handles[0] = mpatches.Patch(color=f'C00', label=f'GPT-4', alpha=0.5)
        legend_handles[2] = mpatches.Patch(color=f'C02', label=f'Claude-2', alpha=0.5)
        legend_handles[3] = mpatches.Patch(color=f'C03', label=f'Claude-1', alpha=0.5)
    elif (experiment == 'TwoStepTask') & (score_name == 'behaviour_score1'):
        plt.ylabel(f'Model-basedness \n($\Delta$ to no prompting)')
        #Add legend handles 5, then blank then the rest
        legend4 = mpatches.Patch(color=f'C04', label=f'LLaMA-2-70', alpha=0.5)
        legend_handles = [legend4] + [mpatches.Patch(color='white', label='')] + legend_handles[5:]

    #! Add two bars after some space for each experiment
    plt.bar(' ', 0, color='white')
    plt.bar('new', agg_means[0], yerr=agg_cis[0], color=f'tab:brown', alpha=0.3)
    plt.bar('new2', agg_means[1], yerr=agg_cis[1], color=f'tab:brown', alpha=0.7)
        
    plt.xticks([])
    # Remove the top and right spines
    sns.despine()
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(top=0.9)
    # Save the plot
    plt.savefig(f'./plots/cot_sb/_{experiment}{score_name}.pdf')

def radar(metrics_dict, agents_to_plot, interest, labels, squash=True):
    '''
    Plot a radar plot for each metric for the chosen llms
    Args:
        metrics_dict (dict): Dictionary with the scores for each engine
        squash (bool, optional): Whether to squash the metrics. Defaults to True.
    '''
    # Process the metrics to be in the right format

    metrics = np.array(list(metrics_dict.values()))
    if squash:
        # Standardize and then squash metrics for each metric
        for i in range(metrics.shape[1]):
            mean = np.mean(metrics[:, i])
            std = np.std(metrics[:, i])
            metrics[:, i] = (metrics[:, i] - mean) / std
            #Make everything from 0 to 1
            metrics[:, i] = metrics[:, i] - np.min(metrics[:, i])
            metrics[:, i] = metrics[:, i] / np.max(metrics[:, i])

    # Define a color palette
    palette = sns.color_palette("hls", 3) # 3 because llm + human + random

    ## Get the metrics for human and random. Also get indices of llms not to plot
    idx_to_remove = []
    # Start with getting the indices
    for i, engine in enumerate(metrics_dict.keys()):
        if engine == 'human':
            human_idx = i
        elif engine == 'random':
            random_idx = i
        elif engine not in agents_to_plot:
            idx_to_remove.append(i)

    # Get the metrics
    human_metrics = metrics[human_idx]
    random_metrics = metrics[random_idx]
    # Remove the metrics from the metrics array and from the metrics_dict
    metrics = np.delete(metrics, [human_idx, random_idx] + idx_to_remove, axis=0)
    metrics_dict.pop('human')
    metrics_dict.pop('random')
    #Remove the agents not to plot
    for idx in sorted(idx_to_remove, reverse=True):
        metrics_dict.pop(list(metrics_dict.keys())[idx])
    
    # Plot the results
    num_vars = len(labels)
    for i, (engine, _) in enumerate(metrics_dict.items()):  
        plt.close()
        fig, ax = plt.subplots(figsize=(3.33, 3.33), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Plot the metrics for the current LLM
        values = metrics[i].tolist()
        ax.fill(angles, values, color=palette[0], alpha=0.25, label=engine)

        # Plot the metrics for human
        values = human_metrics.tolist()
        ax.fill(angles, values, color=palette[1], alpha=0.25, label='human')

        # Plot the metrics for random
        values = random_metrics.tolist()
        ax.fill(angles, values, color=palette[2], alpha=0.25, label='random')

        ax.set_yticklabels([])  # hide radial tick labels (y-axis)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=6)  # adjust the fontsize as needed

        # Create a legend
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol=3, frameon=False, columnspacing=0.25)

        #Size
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'./plots/radarplots/{interest}_{engine}.pdf')

def umap_plot(metrics_dict, feature_of_interest, feature_of_interest_name,  interest='behaviour', standardize=True, name_all=True):
    '''
    Use Umap for each engine. Each dimension is a score. Store the results in a csv file.
    Args:
        metrics_dict (dict): Dictionary with the scores for each engine
        feature_of_interest (list): List with the feature of interest for each engine
        feature_of_interest_name (str): Name of the feature of interest
        standardize (bool, optional): Whether to standardize the metrics. Defaults to True.
    '''
    reducer = umap.UMAP()
    plt.rcParams["figure.figsize"] = (3.33,3.33)

    # Process the metrics to be in the right format
    metrics = np.array(list(metrics_dict.values()))
    if standardize:
        # Standardize the metrics
        scaler = StandardScaler()
        metrics = scaler.fit_transform(metrics)

    # Call the function before UMAP reduction       
    calculate_distances(metrics_dict, feature_of_interest, metrics)

    # Create a color list
    colors = ['tab:red'if boolean == 'Yes' else 'tab:purple' if boolean == 'RLAIF' else 'C00' for boolean in feature_of_interest]
    colors = ['C01' if engine in ['random', 'human'] else color for engine, color in zip(metrics_dict.keys(), colors)]

    # Reduce dimensionality of metrics using UMAP 
    embedding = reducer.fit_transform(metrics)

    # Plot the results
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors,  s=10, alpha=0.7)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    for i, engine in enumerate(metrics_dict.keys()):
        if name_all == False:
            enginestoname = ['gpt-4', 'text-bison', 'human', 'random', 'llama-2-70-chat', 'llama-2-70', 'claude-2'] # Hardcoded for paper
            if engine in enginestoname:
                plt.annotate(engine, (embedding[i, 0], embedding[i, 1]), fontsize='small')            
    plt.gca().set_aspect('equal', 'datalim')

    # Create a legend
    red_patch = mpatches.Patch(color='tab:red', label=feature_of_interest_name, alpha=0.7)
    blue_patch = mpatches.Patch(color='C00', label='No ' + feature_of_interest_name, alpha=0.7)
    purple_patch = mpatches.Patch(color='tab:purple', label='RLAIF', alpha=0.7)
    orange_patch = mpatches.Patch(color='C01', label='Random/Human', alpha=0.7)
    plt.legend(handles=[red_patch, blue_patch, purple_patch, orange_patch],frameon=False,bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol=2, columnspacing=0.5) 
    sns.despine()

    #Size
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'./plots/umap/{interest}{feature_of_interest_name}{"_notstandardized" if not standardize else ""}`.pdf')

    # Call the function after UMAP reduction
    calculate_distances(metrics_dict, feature_of_interest, embedding)

def calculate_distances(metrics_dict, feature_of_interest, embedding):
    '''
    Calculate the distances to the "Human" point for the two groups
    Args:
        metrics_dict (dict): Dictionary with the scores for each engine
        feature_of_interest (list): List with the feature of interest for each engine
        embedding (np.array): Embedding of the metrics
    '''
    # Get the coordinates of the "Human" point
    human_index = list(metrics_dict.keys()).index('human')
    human_coords = embedding[human_index]

    # Calculate the distances to the "Human" point for the two groups
    rlhf_indices = [i for i, feature in enumerate(feature_of_interest) if feature == 'Yes']
    non_rlhf_indices = [i for i, feature in enumerate(feature_of_interest) if feature == 'No']
    rlhf_distances = np.linalg.norm(embedding[rlhf_indices] - human_coords, axis=1)
    non_rlhf_distances = np.linalg.norm(embedding[non_rlhf_indices] - human_coords, axis=1)

    # Calculate the average distances
    avg_rlhf_distance = np.mean(rlhf_distances)
    avg_non_rlhf_distance = np.mean(non_rlhf_distances)

    #print closest one to human for rlhf and non rlhf
    rlhf_closest = np.argmin(rlhf_distances)
    non_rlhf_closest = np.argmin(non_rlhf_distances)
    print(f"Average distance to Humans for LLMs with Use of RLHF: {avg_rlhf_distance}")
    print(f"Average distance to Humans for LLMs without Use of RLHF: {avg_non_rlhf_distance}")
    print(f"Ratio: {avg_rlhf_distance/avg_non_rlhf_distance}")


def merge_all_metrics_and_features(experiments, excluding_agents, llm_df):
    """
    This function reads in the engine features and scores data, and calculates the mean scores and cis for each engine in each experiment.

    Parameters:
    experiments (dict): A dictionary where the keys are the experiment names and the values are lists of score names.
    excluding_agents (list): A list of engines to exclude from the analysis.

    Returns:
    metrics (dict): A dictionary where the keys are the engine names and the values are lists of mean scores for each score name in each experiment.
    metrics_cis (dict): A dictionary where the keys are the engine names and the values are lists of confidence intervals for each score name in each experiment.
    """

    # Get a list of engines to include in the analysis
    including_agents = llm_df['Engine'].unique().tolist()

    # Initialize a dictionary to store the metrics for each engine
    metrics = {engine:[] for engine in including_agents if engine not in excluding_agents}
    metrics_cis = {engine:[] for engine in including_agents if engine not in excluding_agents}

    # Loop over the experiments
    for experiment, scores_names in experiments.items():
        # Read in the scores data for the experiment
        df = pd.read_csv(f"../Experiments/{experiment}/scores_data.csv")

        # Loop over the score names
        for score_name in scores_names:
            # Loop over the engines
            for engine in metrics.keys():
                # Get the scores for the engine
                engine_scores = df[df['engine'] == engine][score_name]

                # Handle special case for the BART experiment and the human engine
                if (experiment == 'BART') & (engine == 'human'):
                    std_men = np.sqrt(7.7**2 + 2**2 + 0.6**2) 
                    std_women = np.sqrt(7.1**2 + 1.8 **2+ 0.5**2)
                    std_total = np.sqrt(std_men ** 2 + std_women ** 2)
                    ci_hum = 1.96 * std_total / np.sqrt(82)
                    metrics_cis['human'].append(ci_hum)
                    if score_name == 'behaviour_score1':
                        men = 32.2*(1+9.1/30) + 7.6*(1+16.5/30) + 1.2*(1+22.2/30)
                        women = 26.4*(1+7.6/30) + 8*(1+15/30) + 1.4*(1+21.7/30)
                        human_pumps = (men+women)/6
                        metrics['human'].append(human_pumps)
                    elif score_name == 'performance_score1':
                        metrics['human'].append((32.2+7.6+1.2+26.4+8+1.4)/6)
                else:
                    # Calculate the mean score for the engine and add it to the metrics dictionary
                    mean_score = np.mean(engine_scores)
                    metrics[engine].append(mean_score)
                    if f'{score_name}_CI' in df.columns:
                        ci = df[df['engine'] == engine][f'{score_name}_CI'].values[0]
                    else:
                        std_score = np.std(engine_scores)
                        ci = 1.96 * std_score / np.sqrt(len(engine_scores))
                    ci = np.nan_to_num(ci)
                    metrics_cis[engine].append(ci)

                    # Check for NaN values
                    if np.isnan(mean_score):
                        print(f"Engine {engine} has NaN values for {score_name} in experiment {experiment}")
                        import ipdb; ipdb.set_trace()

    return metrics, metrics_cis