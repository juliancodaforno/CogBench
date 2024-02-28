import os
import sys
import time
import anthropic
import openai
from ..base_classes import RandomLLM 
from dotenv import load_dotenv
# Import scripts for the different LLMs
from .gpt import GPT3LLM, GPT4LLM
from .anthropic import AnthropicLLM
from .google import GoogleLLM
from .hf import HF_API_LLM

def get_llm(engine, temp, max_tokens, with_suffix=False):
    '''
    Based on the engine name, returns the corresponding LLM object
    '''
    # Initialize flags for step back and CoT as False
    step_back = False
    cot = False

    # Check if the engine name is a prompt engineering technique which 
    # therefore requires more tokens and a boolean flag for signal the appending required at the end
    if engine.endswith('_sb'):
        engine = engine[:-3]
        max_tokens = 350
        step_back = True
    elif engine.endswith('_cot'):
        engine = engine[:-4]
        max_tokens = 350
        cot = True

    # Check which engine is being used and assign the corresponding LLM object with the required parameters (e.g: API keys)
    if engine.startswith("text-davinci") or engine.startswith("text-curie") or engine.startswith("text-babbage") or engine.startswith("text-ada"):
        load_dotenv(); gpt_key = os.getenv("OPENAI_API_KEY")
        llm = GPT3LLM((gpt_key, engine, with_suffix))
    elif engine.startswith("gpt"):
        # load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY{2 if engine == 'gpt-4' else ''}")
        load_dotenv(); gpt_key = os.getenv(f"OPENAI_API_KEY")
        llm = GPT4LLM((gpt_key, engine))
    elif engine.startswith("claude"):
        load_dotenv(); anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        llm = AnthropicLLM((anthropic_key, engine))
    elif engine.startswith("hf") or engine.startswith("llama-2") :
        llm = HF_API_LLM((engine, max_tokens, temp))
    # elif engine.startswith("gemini"):
    #     load_dotenv(); gemini_key = os.getenv("GOOGLE_CREDENTIALS_FILENAME2")
    #     llm = GeminiLLM((gemini_key, engine))
    #     llm.is_gemini = True #See the TODO below
    elif ('bison' in engine):
        load_dotenv(); google_key = os.getenv("GOOGLE_CREDENTIALS_FILENAME2")
        llm = GoogleLLM((google_key, engine))
    else:
        print('No key found')
        llm = RandomLLM(engine)

    #TODO: I am thinking for some models which really are stubborn/tedious processing, to maybe set some flag here in the form is_X = True which could eb recognized in the experiments to mitigate the issues? For now no, to keep things simple and not hardcoded for each LLM.
    # if not hasattr(llm, 'is_X'):
    #     llm.is_X = False
        
    # Set temperature and max_tokens
    llm.temperature = temp
    llm.max_tokens = max_tokens
    llm.step_back = step_back
    llm.cot = cot
    return llm

       