# vector_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from config import DEFAULT_LLM, OLLAMA_DEFAULT_MODEL, VECTOR_DB_DIR, TOP_K, SEARCH_TYPE
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Initializes the HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    """
    Initializes the Ollama LLM.
    """
    return Ollama(model=OLLAMA_DEFAULT_MODEL)

def create_vector_store(document_chunks, persist_directory):
    """
    Creates a Chroma vector store from a list of Document objects and persists it to disk.
    """
    try:
        if not document_chunks:
            logging.warning("No document chunks provided. Skipping vector store creation.")
            return None

        embedding = get_embedding_model()
        vector_db = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        vector_db.persist()
        return vector_db
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return None

def load_vector_store(persist_directory):
    """
    Loads an existing Chroma vector store from disk.
    """
    try:
        embedding = get_embedding_model()
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        return None

def query_vector_store(vector_db, query):
    """
    Queries the vector store with a given query and retrieves relevant documents.
    """
    if vector_db is None:
        return "Error: Vector store is not initialized."

    try:
        llm = get_llm()

        # 1. Retrieval
        if SEARCH_TYPE == "similarity":
            retrieved_docs = vector_db.similarity_search(query, k=TOP_K)
        elif SEARCH_TYPE == "mmr":
            retrieved_docs = vector_db.max_marginal_relevance_search(query, k=TOP_K)
        elif SEARCH_TYPE == "hybrid":
            similarity_docs = vector_db.similarity_search(query, k=TOP_K // 2)
            mmr_docs = vector_db.max_marginal_relevance_search(query, k=TOP_K // 2)
            retrieved_docs = similarity_docs + mmr_docs
        else:
            retrieved_docs = vector_db.similarity_search(query, k=TOP_K)

        if not retrieved_docs:
            return "I'm sorry, I couldn't find any relevant information in the documents."

        # 2. Contextual Compression
        compressor = EmbeddingsFilter(embeddings=get_embedding_model(), similarity_threshold=0.7)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_db.as_retriever())
        compressed_docs = retriever.get_relevant_documents(query=query)

        # 3. Prompt Engineering
        prompt_template = """System: Your name is SkeletaX, a bone health expert who helps people with their questions about bone fractures in Boston, Massachusetts, United States.

        Given the user's question and the following relevant information, provide a clear and concise answer.  Cite the source documents by their title.

        If you cannot answer the question based on the provided information, simply respond with: 'Hmm, I'm not sure about that. Let me see if I can find more information.'

        Context:
        {context}

        User Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        context_text = "\n\n".join([f"Document Title: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in compressed_docs])
        final_prompt = prompt.format(context=context_text, question=query)
        logger.info(f"Final Prompt being sent to LLM:\n{final_prompt}")
        response = llm(final_prompt)
        logger.info(f"LLM Response:\n{response}")
        return response

    except Exception as e:
        logging.error(f"Error querying vector store: {e}")
        return f"Sorry, there was an error processing your query: {e}"
