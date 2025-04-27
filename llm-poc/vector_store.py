import os
import shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.llms import Ollama  # Import Ollama for potential direct use
from config import DEFAULT_LLM, OPENAI_DEFAULT_MODEL, OLLAMA_DEFAULT_MODEL
from openai_llm import OpenAILLM
from ollama_llm import OllamaLLM

# --- Vector Store Management ---
def create_vector_store(chunks, persist_dir: str):
    """Create and persist a Chroma vector store using HuggingFace embeddings."""
    if os.path.exists(persist_dir):
        print(f"Removing existing vector store from {persist_dir}")
        shutil.rmtree(persist_dir)

    print(f"Total chunks received for vector store: {len(chunks)}")
    if chunks:
        print(f"Example chunk: {chunks[0].page_content[:300]}")

    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print(f"Building and saving the new vector store at {persist_dir} with HuggingFace embeddings...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        return vector_db

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def load_vector_store(persist_dir: str):
    """Loads an existing Chroma vector store using HuggingFace embeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        print(f"Vector store loaded from {persist_dir}")
        return vector_db
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# --- Question Answering with LLM (Manual Prompting) ---
def query_vector_store(vector_db, query, k_retriever=3, score_threshold=0.3):
    """Queries the vector store and generates a natural language response using an LLM with manual prompt construction."""
    if vector_db is None:
        print("Error: Vector store not initialized.")
        return "Error: Vector store not initialized."

    llm_engine = None
    if DEFAULT_LLM == "openai":
        llm_engine = OpenAILLM(model_name=OPENAI_DEFAULT_MODEL)
    elif DEFAULT_LLM == "ollama":
        llm_engine = OllamaLLM(model_name=OLLAMA_DEFAULT_MODEL)
    else:
        raise ValueError(f"Unsupported LLM type: {DEFAULT_LLM}")

    search_kwargs = {
        "k": k_retriever,
        "score_threshold": score_threshold
    }

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )

    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    try:
        response = llm_engine.generate_response(question=query, context=context)
        return response
    except Exception as e:
        print(f"Error during LLM query: {e}")
        return "Sorry, there was an error processing your request."