This folder contains scripts for different LLMs. Each script implements a subclass of the `LLM` class defined in `../base_classes.py`. The `get_llm()` in `llms.py` is the function that is used to query the right LLM based on the engine name for each experiments (see `../Experiments` folder).

## Structure

Each LLM script contains a class that inherits from the `LLM` base class. The class implements a `_generate()` method, which is used to standardize the query and response format for all LLMs.

## Adding a New LLM

If you want to add your own LLM, you just need to create a new script in this folder. Your script should define a class that inherits from the `LLM` base class and implements the `_generate()` method. We actually advise users to create their own script with LLM subclass and using the other scripts only as a reference.

## Engine Names

The engine name is used to determine which LLM to use. Some engine names can be processed by these scripts to query the right API or use the right prompt engineering technique.

For example, all Hugging Face models are recognized by prepending `hf_` to the engine name. If you want to use the Chain of Thoughts or Take a Step Back techniques, you can add `_cot` or `_sb` as a suffix to the engine name, respectively. You can see how these engineering techniques are processed in the `LLM` class defined in `base_classes.py`.

Please note that if the engine name is not recognized, a `NotImplementedError` will be raised.

Feel free to contribute to this folder by adding your own LLM scripts or improving the existing ones. 