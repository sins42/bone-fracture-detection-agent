from langchain_ollama import OllamaLLM as LangchainOllamaLLM  
from llm_interface import LLM
from prompt_templates import PROMPT_TEMPLATE 

class OllamaLLM(LLM):
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.llm = LangchainOllamaLLM(model=model_name) 
        self.prompt_template = PROMPT_TEMPLATE 

    def generate_response(self, question: str, context: str) -> str:
        prompt = self.prompt_template.format(context=context, question=question)
        print("--- OllamaLLM generate_response ---")
        print(f"Full Prompt being sent to Ollama:\n{prompt}")
        print("--- End of Prompt ---")
        try:
            response = self.llm(prompt)
            print(f"Ollama Response:\n{response}")
            return response
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "Sorry, there was an error processing your request with Ollama."