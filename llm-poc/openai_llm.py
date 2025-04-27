from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from llm_interface import LLM
from prompt_templates import PROMPT_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

class OpenAILLM(LLM):
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm: BaseChatModel = ChatOpenAI(model_name=model_name)  # Use ChatOpenAI
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

    def generate_response(self, question: str, context: str) -> str:
        prompt = self.prompt_template.format_messages(context=context, question=question)
        try:
            response = self.llm.invoke(prompt)  # Use .invoke() for ChatOpenAI
            return response.content  # Access the content of the ChatMessage
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return "Sorry, there was an error processing your request with OpenAI."