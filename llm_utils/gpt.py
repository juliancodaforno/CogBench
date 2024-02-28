import openai
import time
import sys
from ..base_classes import LLM 

class GPT3LLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        self.gpt_key, self.engine, self.suffix = llm_info
        openai.api_key = self.gpt_key

    def _generate(self, text, temp, max_tokens):
        for iter in range(10):
            try:
                response = openai.Completion.create(
                    engine = self.engine,
                    prompt = text,
                    max_tokens = max_tokens,
                    temperature = temp,
                    suffix =self.suffix if self.suffix else None
                )
                return response.choices[0].text.strip()
            except:
                print(f"Error in GPT3: {sys.exc_info()[0]}")
                time.sleep(3**iter)

class GPT4LLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        self.gpt_key, self.engine = llm_info
        openai.api_key = self.gpt_key

    def _generate(self, text, temp, max_tokens):
        text = [{"role": "user", "content": text}]  
        time.sleep(1) # to avoid rate limit error which happens a lot for gpt4
        for iter in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model = self.engine,
                    messages = text,
                    max_tokens = max_tokens,
                    temperature = temp
                )
                output = response.choices[0].message.content

                # print(text, output, '\n\n\n\n')
                return output
            except:
                print(sys.exc_info()[0])
                time.sleep(3**iter)
                if iter == 5:
                    import ipdb; ipdb.set_trace()