import os
import vertexai
import time
import sys
from vertexai.preview.generative_models import GenerativeModel, ChatSession
from vertexai.preview.language_models import TextGenerationModel, ChatModel 
from ..base_classes import LLM

class GoogleLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        google_key, engine = llm_info

        # this is a key file for a service account, which only has the role "Vertex AI User"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'../{google_key}.json'
        # vertexai.init(project="XX", location="XX") #TODO: Add the project and location
        self.model = TextGenerationModel.from_pretrained(engine)

    def _generate(self, text, temp, max_tokens):
        time.sleep(12.01) #Due to rate limits of 5 queries per minute
        try:
            text = text.rstrip()
            response = self.model.predict(
                text,
                temperature=temp,  
                max_output_tokens=max_tokens, 
            )
            output = response.text

            print(text, response.text, '\n\n\n\n')
        except:
            print(f"Error in google is: {sys.exc_info()[0]}, try again once....")
            time.sleep(100)
            response = self.model.predict(
                            text,
                            temperature=temp,
                            max_output_tokens=max_tokens,
                        )
            output = response.text
        return output
    
#TODO: Work in progress..
# class GeminiLLM(LLM):
#     def __init__(self, llm_info):
#         super().__init__(llm_info)
#         google_key, engine = llm_info
#         self.Q_A =  ('\n\nUser:', '\n\nAI:')

#         # this is a key file for a service account, which only has the role "Vertex AI User"
#         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'../{google_key}.json'
#        # vertexai.init(project="XX", location="XX") #TODO: Add the project and location
#         self.model = GenerativeModel(engine)


#     def _generate(self, text, temp, max_tokens):
#         text = text.rstrip()
#         time.sleep(12.01) #Due to rate limits
#         config = {
#         "max_output_tokens": max_tokens + 2, #+2 here because in metacognition I get rid of the first two tokens '0.' because it causes some problems where gemini just repeats the prompt and then gives the output.
#         "temperature": temp,
#         "top_p": 1
#         }
        
#         from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold, ResponseBlockedError
#         safety_settings = {
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         }

#         response = self.model.generate_content(
#             text, generation_config=config, safety_settings=safety_settings
#         )
        
#         print(response.text)
#         import ipdb; ipdb.set_trace()

#         return response.text