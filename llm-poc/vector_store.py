# vector_store.py
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Standardized import
from langchain.llms import Ollama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from config import DEFAULT_LLM, OLLAMA_DEFAULT_MODEL, VECTOR_DB_DIR, TOP_K, SEARCH_TYPE, MAIN_PROMPT_TEMPLATE, STAGING_PROMPT_TEMPLATE
import logging
import os
from typing import List, Optional
from langchain.docstore.document import Document 

# Set up logging
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Initializes the HuggingFace embedding model.
    """
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise

def get_llm():
    """
    Initializes the Ollama LLM.
    """
    try:
        return Ollama(model=OLLAMA_DEFAULT_MODEL)
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {e}")
        raise

def create_vector_store(document_chunks: List[Document], persist_directory: str) -> Optional[Chroma]:
    """
    Creates a Chroma vector store from a list of Document objects and persists it to disk.

    Args:
        document_chunks (List[Document]): List of documents to add to the vector store.
        persist_directory (str): Directory to persist the vector store.

    Returns:
        Chroma: The created Chroma vector store or None if there was an error.
    """
    try:
        if not document_chunks:
            logger.warning("No document chunks provided. Skipping vector store creation.")
            return None

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        embedding = get_embedding_model()
        logger.info(f"Creating vector store with {len(document_chunks)} documents...")
        
        # Log a sample of the documents for debugging
        if document_chunks:
            logger.info(f"Sample document: {document_chunks[0].page_content[:100]}...")
            logger.info(f"Sample metadata: {document_chunks[0].metadata}")
        
        vector_db = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        vector_db.persist()  # Explicitly persist to disk
        logger.info(f"Vector store created and persisted at {persist_directory}")
        return vector_db
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def load_vector_store(persist_directory: str) -> Optional[Chroma]:
    """
    Loads an existing Chroma vector store from disk.

    Args:
        persist_directory (str): Directory where the vector store is persisted.

    Returns:
        Chroma: The loaded Chroma vector store or None if there was an error.
    """
    try:
        if not os.path.exists(persist_directory):
            logger.error(f"Vector store directory does not exist: {persist_directory}")
            return None
            
        embedding = get_embedding_model()
        logger.info(f"Loading vector store from {persist_directory}...")
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        
        # Test if the vector store has documents
        collection_count = vector_db._collection.count()
        logger.info(f"Loaded vector store with {collection_count} documents")
        
        if collection_count == 0:
            logger.warning("Vector store is empty. Consider rebuilding it.")
            
        return vector_db
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

def extract_response(response_text: str) -> str:
    """
    Extract and clean the response from the LLM output.
    
    Args:
        response_text (str): The raw response from the LLM
        
    Returns:
        str: Cleaned response
    """
    # Extract answer part if present
    if "Answer:" in response_text:
        response_text = response_text.split("Answer:")[-1].strip()
    
    # Remove thinking sections if present
    if "<think>" in response_text and "</think>" in response_text:
        parts = response_text.split("<think>")
        for part in parts:
            if "</think>" in part:
                thinking, rest = part.split("</think>", 1)
                response_text = response_text.replace(f"<think>{thinking}</think>", "")
    
    return response_text.strip()

def query_vector_store(vector_db: Optional[Chroma], query: str) -> str:
    """
    Queries the vector store with a given query and retrieves relevant documents.
    Implements a two-stage prompt.

    Args:
        vector_db (Chroma): The Chroma vector store to query.
        query (str): The user query.

    Returns:
        str: The answer to the query.
    """
    if vector_db is None:
        return "Error: Vector store is not initialized. Please try again later."

    try:
        llm = get_llm()

        # Step 1: Staging Prompt to reformulate the query
        logger.info(f"Original query: {query}")
        staging_prompt = PromptTemplate(template=STAGING_PROMPT_TEMPLATE, input_variables=["user_query"])
        staged_query_prompt = staging_prompt.format(user_query=query)
        logger.info(f"Staged query prompt being sent to LLM")
        
        staged_response = llm(staged_query_prompt).strip()
        logger.info(f"LLM staged response: {staged_response}")
        
        # Check if the query is off-topic
        if staged_response == "OFF_TOPIC":
            return "I can only answer questions about elbow, hand, or shoulder fractures. Please ask a question related to these topics."
        
        # Use the staged response for document retrieval
        effective_query = staged_response
        
        # Step 2: Document Retrieval with different strategies
        if SEARCH_TYPE == "similarity":
            retrieved_docs = vector_db.similarity_search(effective_query, k=TOP_K)
        elif SEARCH_TYPE == "mmr":
            retrieved_docs = vector_db.max_marginal_relevance_search(effective_query, k=TOP_K)
        elif SEARCH_TYPE == "hybrid":
            similarity_docs = vector_db.similarity_search(effective_query, k=TOP_K // 2)
            mmr_docs = vector_db.max_marginal_relevance_search(effective_query, k=TOP_K // 2)
            retrieved_docs = similarity_docs + mmr_docs
        else:
            retrieved_docs = vector_db.similarity_search(effective_query, k=TOP_K)

        if not retrieved_docs:
            return "I couldn't find any relevant information in my knowledge base about your question. Please try rephrasing your question about elbow, hand, or shoulder fractures."

        # Step 3: Context Compilation
        context_text = ""
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Unknown')
            heading = doc.metadata.get('heading', 'General Information')
            context_text += f"Document {i+1}: {source} - {heading}\n{doc.page_content}\n\n"

        # Step 4: Generate Response
        main_prompt = PromptTemplate(template=MAIN_PROMPT_TEMPLATE, input_variables=["context", "question"])
        final_prompt = main_prompt.format(context=context_text, question=query)
        logger.info(f"Sending final prompt to LLM")
        
        response = llm(final_prompt).strip()
        logger.info(f"Raw LLM response received")
        
        # Clean up the response
        clean_response = extract_response(response)
        logger.info(f"Clean response: {clean_response[:100]}...")
        
        return clean_response

    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return "I'm having trouble processing your question right now. Please try again or rephrase your question about bone fractures."