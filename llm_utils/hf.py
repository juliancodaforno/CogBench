import torch
import transformers
from huggingface_hub import HfApi, HfFolder, InferenceApi
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ..base_classes import LLM 

class HF_API_LLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, temperature = llm_info

        # Adapt engine name to Hugging Face API. Here I prepend hf_ to the engine name for all the hf models.
        if engine.startswith('hf_'):
            engine = engine.split("hf_")[1] # remove the "hf_" part
        padtokenId = 50256 # Falcon needs that to avoid some annoying warning
        if engine.startswith("falcon"):
            engine = "tiiuae/" + engine 
        elif engine.startswith("mpt"):
            engine = "mosaicml/" + engine
        elif engine.startswith("vicuna"):
            engine = "lmsys/" + engine
        elif engine.startswith("koala"):
            engine = 'TheBloke/' + engine
        elif ('longlora' in engine) or ('Alpaca' in engine):
            engine = 'Yukang/' + engine
        elif 'CodeLlama' in engine:
            engine = 'codellama/' + engine + '-hf'
        elif engine.startswith('llama-2'):
            #Change llama-2-* to meta-llama/Llama-2-*b-hf
            if 'chat' in engine:
                engine = 'meta-llama/L' +engine[1:].replace('-chat', '') + 'b-chat-hf'
            else:
                engine = 'meta-llama/L' +engine[1:] + 'b-hf'
        else:
            print("Wrong engine name for HF API LLM")
            raise NotImplementedError

        print(engine)
        try:   
            tokenizer = AutoTokenizer.from_pretrained(engine)
            # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')
            self.pipe = transformers.pipeline(
                "text-generation",
                model=engine,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                pad_token_id=padtokenId,
                max_new_tokens=max_tokens
            )
        except:
            # For Longlora models don't include tokenizer
            self.pipe = transformers.pipeline(
                "text-generation",
                model=engine,
                # tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                pad_token_id=padtokenId,
                max_new_tokens=max_tokens
            )
        # Adapt pipeline to set temperature to 0
        self.pipe.model.config.temperature = temperature +1e-6
    
    def _generate(self, text, temp, max_tokens):
        response = self.pipe(text)[0]['generated_text'][len(text):]
        return response
