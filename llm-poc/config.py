# config.py
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_key(dotenv_path="../.env"):
    """Loads the API key from a .env file."""
    load_dotenv(dotenv_path=dotenv_path)
    return None

# Define project-level constants
VECTOR_DB_DIR = os.path.join(os.getcwd(), "/SkeletaX_chroma_db")
TEMP_PDF_DIR = "../temp_pdfs"
DATA_DIR = "../data/bone_fractures_RAG_data.zip"
DEFAULT_LLM = "ollama"
OLLAMA_DEFAULT_MODEL = "deepseek-r1:1.5b"
CHUNK_SIZE = 500  # Reduced chunk size for more granular chunks
CHUNK_OVERLAP = 100  # Reduced overlap
TOP_K = 5
SEARCH_TYPE = "similarity"  # You can also experiment with "mmr" or "hybrid"