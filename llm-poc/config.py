# config.py
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_key(dotenv_path="../.env"):
    """Loads the API key from a .env file."""
    load_dotenv(dotenv_path=dotenv_path)
    # No need to check for OpenAI key anymore
    return None  # Return None since we're not using it

# Define project-level constants
VECTOR_DB_DIR = os.path.join(os.getcwd(), "/SkeletaX_chroma_db")
TEMP_PDF_DIR = "../temp_pdfs"
DATA_DIR = "../data/bone_fractures_RAG_data.zip"
DEFAULT_LLM = "ollama"  # Force Ollama
OLLAMA_DEFAULT_MODEL = "deepseek-r1:1.5b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
SEARCH_TYPE = "similarity"