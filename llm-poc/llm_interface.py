# llm_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLM(ABC):
    """
    Abstract base class for language model implementations.
    Different LLM backends should implement this interface.
    """
    
    @abstractmethod
    def generate_response(self, question: str, context: str) -> str:
        """
        Generate a response to a user question using the provided context.
        
        Args:
            question (str): The user's question
            context (str): Relevant context information to use in answering
            
        Returns:
            str: The generated response
        """
        pass
        
    def _clean_response(self, text: str) -> str:
        """
        Clean up the response from the LLM.
        
        Args:
            text (str): The raw response from the LLM
            
        Returns:
            str: Cleaned response
        """
        return text.strip()