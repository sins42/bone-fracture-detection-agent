# ollama_llm.py
from langchain_ollama import OllamaLLM as LangchainOllamaLLM
from llm_interface import LLM
from prompt_templates import PROMPT_TEMPLATE
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """
    Wrapper around the Ollama LLM.
    """
    def __init__(self, model_name="deepseek-r1:1.5b"):
        try:
            self.llm = LangchainOllamaLLM(model=model_name)
            self.prompt_template = PROMPT_TEMPLATE
            logger.info(f"Ollama LLM initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise  # Re-raise the exception to prevent the program from continuing

    def generate_response(self, question: str, context: str) -> str:
        """
        Generates a response to a user question using the Ollama LLM.

        Args:
            question (str): The user's question.
            context (str): The relevant context retrieved from the vector store.

        Returns:
            str: The LLM's response, or an error message.
        """
        prompt = self.prompt_template.format(context=context, question=question)
        logger.info("--- OllamaLLM generate_response ---")
        logger.info(f"Full Prompt being sent to Ollama:\n{prompt}")
        try:
            response = self.llm(prompt)
            logger.info(f"Ollama Response:\n{response}")
            return response
        except Exception as e:
            error_message = f"Error calling Ollama: {e}"
            logger.error(error_message)
            return "Sorry, there was an error processing your request with Ollama."
