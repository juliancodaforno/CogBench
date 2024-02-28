import anthropic
import sys
import time
from ..base_classes import LLM 

class AnthropicLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        self.anthropic_key, self.engine = llm_info
        self.Q_A =  (anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT)

    def _generate(self, text, temp, max_tokens):
        c = anthropic.Client(self.anthropic_key)
        count = 0
        while count < 10:
            try: 
                time.sleep(3**count - 1)
                response = c.completion(
                    prompt = anthropic.HUMAN_PROMPT + text.rstrip(),
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model=self.engine,
                    temperature=temp,
                    max_tokens_to_sample=max_tokens,
                )['completion']
                return response
            except:
                print(f"Error in anthropic for count {count}: {sys.exc_info()[0]}")
                count += 1
                continue