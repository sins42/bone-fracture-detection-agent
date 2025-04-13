from abc import ABC, abstractmethod
from typing import Dict, Any

class LLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: str) -> str:
        pass