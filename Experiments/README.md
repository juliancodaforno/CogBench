This repository contains a set of cognitive psychology experiments for querying  LLMs as well as their respective behavioral and performance analysis. Each subfolder represents one of the seven cognitive psychology experiment explained in the paper and has the same following structure as the one below. 

# Folder Structure
- query.py: This script is used to query LLMs for the chosen experiment. It stores the output in the data/ subfolder.
- data/: This folder stores the output of the experiments in the format <enginename>.csv. Each CSV file corresponds to a specific LLM engine.
- envs/: If applicable, this subfolder contains environment files used during the experiment setup in query.py.
- store.py: This script processes the experiment results from the CSV files in data/. It retrieves relevant metrics for each LLM using analysis from cognitive psychology and compiles them into scores_data.csv.
- scores_data.csv: The aggregated results file containing behavioral and performance metrics for all engines.
## Usage
**1. Experiment Setup**:
- Run the query.py script with appropriate parameters to collect data from different LLMs. Flags sometimes vary but some common ones include:
    - --engine: The name of the LLM engine to query. For example, gpt-2, t5, bart, etc and the names depend on how you setup the get_llm() function in ../llm_utils/llms.py script. If --engines is not given then the default is random for the run of random agent on the experiment
    - --num_runs: The number of runs to execute for each engine on that experiment. The default changes depending on the experiment.
    - --version (optional): The version of the experiment to run. By default, this is set to the original version of the experiment. However, you can specify a different version number to run a modified version of the experiment. For example, if you set --version_number to 2, the experiment might change the prompt or the experiment rewards. The data for this version will be stored in a new folder, `dataV2/`. Check the `RestlessBandit` experiment for an example of changing the experiment mean rewards if --version_number is set to 2. This option has not been used almost anywhere in the experiments but just there  to flexibly run different versions of the same experiment which might help for improvements or debugging.
    - --debug (optional): If set, the script will print debug information to the console. This hits a debugger after each query of an LLM and prints the input and raw output of the LLM (before post-processing - so an answer being '2:' or '2(') for a question about option 1 and 2 is rightly processed afterwards.). The output will be in the format "Answer of LLM: '<raw output>'". This feature allows users to verify the input provided to the LLMs and understand the structure of the experiment. It also helps ensure that the output from the LLM is reasonable and as expected. Indeed, it can be used to address potential issues with LLMs. Consider this as an illustrative example: some smaller, erratic models might consistently start any response with a fixed phrase, say "The answer is". Given that the benchmark is typically designed for outputs with a maximum of two tokens, there may not be any meaningful content left to process in such cases. This is just a hypothetical scenario to highlight that additional engineering may sometimes be required to ensure that the outputs from the new introduced LLMs are processed correctly and effectively.
- Store the output in the data/ subfolder using the format <enginename>.csv.
- **2. Analysis**:
Execute the store.py script to process the experiment results.
Specify the LLM engines to analyze using the --engines flag. If no engines are provided, all engines in data/ will be analyzed by default.
The script computes various metrics beyond those listed in the original paper, allowing flexibility for users to add new metrics.
NB: All analysis are relatively fast except for the InstrumentalLearning experiment since the metrics are retrieved using the best fits of rescorla-wagner models and therefore can take a while to run. It is advised for this to really set only the engines you want to analyze and still be patient.
- **3. Metrics and Interpretation**:
The `scores_data.csv` file holds the metrics used by CogBench to evaluate and compare the performance and behavior of various LLMs. These files are accessed in the `../Analysis` directory for the analysis of LLMs across CogBench (the ones done for the paper).
While the paper may only list a subset of these metrics, the `scores_data.csv` file typically contains additional ones. This is to demonstrate to users how they can customize their analysis of a chosen experiment by just changing the store.py file and adding some analysis and metric. Users can add new metrics or modify existing ones as needed to suit their specific requirements.

# Example Usage
`python3 query.py --engine claude-2 gpt-4 --debug`
`python3 store.py --engines claude-2 gpt-4`

Contributing
Contributions are welcome! If you have additional metrics or improvements to the analysis process, feel free to submit a pull request.