import os
from dotenv import load_dotenv

def load_api_key(dotenv_path="../.env"):
    load_dotenv(dotenv_path=dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    default_llm = os.getenv("DEFAULT_LLM", "ollama") # Get DEFAULT_LLM, default to "ollama"

    # We don't need to check for OpenAI API key if we are defaulting to Ollama
    if default_llm != "ollama" and openai_api_key is None:
        print("Error: OPENAI_API_KEY not found in .env file when using OpenAI.")
        exit()
    # We still set the environment variable in case it's used elsewhere (even if None)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return openai_api_key

# Define project-level constants
VECTOR_DB_DIR = os.path.join(os.getcwd(), "/SkeletaX_chroma_db")
TEMP_PDF_DIR = "../temp_pdfs"
DATA_DIR = "../data/bone_fractures_RAG_data.zip"
#DEFAULT_LLM = "openai"
DEFAULT_LLM = "ollama"
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo"
OLLAMA_DEFAULT_MODEL = "deepseek-r1:1.5b"