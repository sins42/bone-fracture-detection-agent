# ollama_llm.py
from langchain_ollama import OllamaLLM as LangchainOllamaLLM
from llm_interface import LLM
from config import OLLAMA_DEFAULT_MODEL, MAIN_PROMPT_TEMPLATE
import logging
from typing import Dict, Any, Optional

# Initialize logger
logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """
    Wrapper around the Ollama LLM.
    """
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Ollama LLM wrapper.
        
        Args:
            model_name (str, optional): The name of the Ollama model to use.
                                        Defaults to the value in config.py.
        """
        try:
            self.model_name = model_name or OLLAMA_DEFAULT_MODEL
            self.llm = LangchainOllamaLLM(
                model=self.model_name,
                temperature=0.2,  # Lower temperature for more focused responses
                repeat_penalty=1.1,  # Slightly penalize repetition
                stop=["<|im_end|>", "Human:", "User:"]  # Stop sequences
            )
            self.prompt_template = MAIN_PROMPT_TEMPLATE
            logger.info(f"Ollama LLM initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            raise

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
        
        logger.info(f"Generating response with Ollama model: {self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            # Log the first 100 characters of the prompt for debugging
            logger.debug(f"Prompt preview: {prompt[:100]}...")
            
            # Call the Ollama model
            response = self.llm(prompt)
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Log the first 100 characters of the response for debugging
            logger.debug(f"Response preview: {response[:100]}...")
            
            return response
            
        except Exception as e:
            error_message = f"Error calling Ollama: {e}"
            logger.error(error_message)
            return "I'm having trouble processing your request right now. Please try again in a moment."

    def _clean_response(self, text: str) -> str:
        """
        Clean up the response from the LLM.
        
        Args:
            text (str): The raw response from the LLM
            
        Returns:
            str: Cleaned response
        """
        # Remove any system or instruction text that might have leaked
        if "System:" in text:
            text = text.split("System:", 1)[0]
            
        # Remove thinking sections if present
        if "<think>" in text and "</think>" in text:
            parts = text.split("<think>")
            for part in parts:
                if "</think>" in part:
                    thinking, rest = part.split("</think>", 1)
                    text = text.replace(f"<think>{thinking}</think>", "")
        
        # Remove "Answer:" if present
        if "Answer:" in text:
            text = text.split("Answer:", 1)[1]
            
        # Remove any "User Question:" prefix if it got included
        if "User Question:" in text:
            text = text.split("User Question:", 1)[0]
            
        return text.strip()