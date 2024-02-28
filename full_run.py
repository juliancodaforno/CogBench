"""
This script is used to run the entire benchmark for a chosen LLM. 
It runs all the experiments, stores the required values, and prints and plots the main metrics.

Usage:
    python3 full_run.py --engine <LLM> [--compare_with <models>]

Arguments:
    --engine: The LLM to run the benchmark on. This is a required argument.
    --compare_with: The models to compare against. This is an optional argument. If not provided, it defaults to ['gpt-4', 'claude-2'].
    --only_analysis: If set, only run the analysis and skip the experiment running and storing steps. This is an optional argument.

Functions:
    run_benchmark(engine): Runs the benchmark for the specified LLM.

Note:
    The fitting of scores is generally fast for all experiments, except for the InstrumentalLearning experiment which can be very slow. 
    Please be patient when running this experiment.
    After the analysis, a summary table is printed with the scores for the chosen agent, as well as the human & random agents and the reference scores for the models specified with the `--compare_with` flag. 
    The performance and behavior normalized scores versus the models specified with the `--compare_with` flag are also plotted. 
    The plots are saved in the `./Analysis/plots/phenotypes/full_runs{interest}.pdf` directory.
"""

import os
import argparse
import subprocess

def run_benchmark(engine):
    """
    This function runs the benchmark for the specified LLM. It runs all the experiments, stores the required values, 
    and prints and plots the main metrics.

    Arguments:
        engine: The LLM to run the benchmark on.

    Returns:
        None
    """
    experiments_dir = './Experiments'
    analysis_dir = './Analysis'

    if not args.only_analysis:
        # Get all the experiment folders
        experiment_folders = [f.path for f in os.scandir(experiments_dir) if f.is_dir()]

        for folder in experiment_folders:
            # Run query.py and store.py for each experiment
            os.chdir(folder)
            print(f'Running experiment {os.path.basename(folder)}')
            subprocess.run(['python3', 'query.py', '--engines', engine])
            print(f'Storing the behavioral scores for experiment {os.path.basename(folder)}')
            subprocess.run(['python3', 'store.py', '--engines', engine])
            os.chdir('../..')  # Go back to the root directory

    # Run phenotype_comp.py in the Analysis folder
    os.chdir(analysis_dir)
    print(f'Behaviour scores:')
    subprocess.run(['python3', 'phenotype_comp.py', '--models', args.engine] + args.compare_with + 
                   ['--interest', 'behaviour', '--store_id', 'full_run', '--print_scores', '--print_scores_for', 'human', 'random', args.engine] + args.compare_with)
    print(f'Performance scores:')
    subprocess.run(['python3', 'phenotype_comp.py', '--models', args.engine] + args.compare_with + 
                   ['--interest', 'performance', '--store_id', 'full_run', '--print_scores', '--print_scores_for', 'human', 'random', args.engine] + args.compare_with) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the entire benchmark for a chosen LLM.')
    parser.add_argument('--engine', type=str, required=True, help='The LLM to run the benchmark on.')
    parser.add_argument('--compare_with', type=str, nargs='+', default=['gpt-4', 'claude-2'], help='The models to compare against.')
    parser.add_argument('--only_analysis', action='store_true', help='If set, only run the analysis and skip the experiment running and storing steps.')
    args = parser.parse_args()

    run_benchmark(args.engine)