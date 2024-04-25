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
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        count = 0
        import ipdb ; ipdb.set_trace()
        while count < 10:
            try: 
                time.sleep(3**count - 1)
                structured_text = self.structure_text(text.rstrip())
                response = client.messages.create(
                                            model=self.engine,  # Model name
                                            max_tokens=max_tokens,                 # Maximum number of tokens to generate
                                            temperature=temp,                 # Sampling temperature
                                            messages=structured_text
                                        ).content[0].text
                return response
            except:
                print(f"Error in anthropic for count {count}: {sys.exc_info()[0]}")
                if count > 3:
                    import ipdb ; ipdb.set_trace()
                count += 1
                continue

    def structure_text(self, text):
        """
        This function restructures a given text from a Q&A format to a format suitable for the Anthropic API.

        The input text should be in the format:
        '...self.Q_A[0]...self.Q_A[1]...self.Q_A[0]...self.Q_A[1]...'
        where self.Q_A[0] and self.Q_A[1] are placeholders for the question and answer prefixes respectively.

        The function transforms the text into a list of dictionaries, where each dictionary represents a message.
        Each message has a 'role' key which can be 'user' or 'assistant', and a 'content' key which holds the text of the message.

        For example, the text 'Q: What is your name? A: My name is AI.' with self.Q_A[0] = 'Q:' and self.Q_A[1] = 'A:'
        would be transformed into:
        [{'role': 'user', 'content': ' What is your name? '}, {'role': 'assistant', 'content': ' My name is AI.'}]

        Parameters:
        text (str): The input text in Q&A format.

        Returns:
        new_messages (list): The transformed text as a list of message dictionaries.
        """
        #Split for all 'Q:'
        text = text.split(self.Q_A[0])
        messages = [{"role": "user", "content": text[i]} for i in range(0, len(text))]
        new_messages = []

        #Iterate over the messages split by 'Q:'
        for user_texts in messages:
            # If within user_texts there is 'A:' then split it and add it to the messages with role assistant
            if self.Q_A[1] in user_texts['content']:
                user_text, assistant_text = user_texts['content'].split(self.Q_A[1])
                if (len(new_messages) != 0) and (new_messages[-1]['role'] == 'user'):
                    new_messages[-1]['content'] += self.Q_A[0] + user_text
                else:
                    new_messages.append({"role": "user", "content": user_text})
                new_messages.append({"role": "assistant", "content": assistant_text})
            else:
                if len(new_messages) == 0:
                    new_messages.append(user_texts)
                else:
                    #Concatenate the last user message with the current one
                    new_messages[-1]['content'] += self.Q_A[0] + user_texts['content']

        return new_messages
