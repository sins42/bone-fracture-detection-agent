# main.py
# Module imports
import logging
import os
import shutil
from typing import List, Optional
import sys

from langchain.docstore.document import Document
from config import (
    load_api_key,
    VECTOR_DB_DIR,
    TEMP_PDF_DIR,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_processing import extract_zip, process_single_pdf
from text_processing import create_chunks_with_structured_data
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("skeletax.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_and_process_pdfs(zip_path: str) -> List[Document]:
    """
    Processes a zipped folder of PDFs, extracts text, creates chunks with structured
    data, and adds source file information to the metadata.

    Args:
        zip_path (str): Path to the zipped folder of PDFs.

    Returns:
        list: A list of Document objects, or an empty list on error.
    """
    try:
        logger.info(f"Processing PDF zip file: {zip_path}")
        
        # Extract PDFs from the zip file
        pdf_files = extract_zip(zip_path, extract_to=TEMP_PDF_DIR)
        if not pdf_files:
            logger.warning("No PDF files found in the zip archive.")
            return []

        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Process each PDF file
        for pdf_path in pdf_files:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract structured content from the PDF
            structured_sections = process_single_pdf(pdf_path)
            if not structured_sections:
                logger.warning(f"No content extracted from {pdf_path}. Skipping.")
                continue
                
            logger.info(f"Created {len(structured_sections)} sections from {pdf_path}")
            
            # Create semantic chunks from the structured sections
            pdf_chunks = create_chunks_with_structured_data(structured_sections, text_splitter)
            logger.info(f"Created {len(pdf_chunks)} chunks from {pdf_path}")
            
            # Add the chunks to our collection
            all_chunks.extend(pdf_chunks)
        
        logger.info(f"Total chunks created from all PDFs: {len(all_chunks)}")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error in load_and_process_pdfs: {e}")
        return []


def main():
    """
    Main function to load PDFs, create/load a vector store, and handle querying.
    """
    try:
        # Initialize API key if needed
        load_api_key()
        
        # Path to the zip file containing the PDFs
        zip_file_path = DATA_DIR
        
        # Force recreation of the vector store - remove this in production if you want to keep the existing store
        if os.path.exists(VECTOR_DB_DIR):
            logger.info(f"Removing existing vector store at {VECTOR_DB_DIR} to force recreation...")
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
            
        # Load the vector store or create a new one
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            logger.info(f"Loading existing vector store from {VECTOR_DB_DIR}...")
            vector_db = load_vector_store(VECTOR_DB_DIR)
            if vector_db is None:
                logger.error("Failed to load vector store. Creating a new one.")
                # Fall through to create a new vector store
            else:
                logger.info("Vector store loaded successfully.")
                
        # If we couldn't load the vector store or it doesn't exist, create a new one
        if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR) or vector_db is None:
            logger.info("Processing PDFs and building a new vector store...")
            document_chunks = load_and_process_pdfs(zip_file_path)
            
            if not document_chunks:
                logger.error("No document chunks were generated. Cannot create vector store.")
                return
                
            logger.info(f"Total chunks created: {len(document_chunks)}")
            
            # Log some sample chunks for debugging
            if document_chunks:
                logger.info(f"Example chunk content: {document_chunks[0].page_content[:100]}...")
                logger.info(f"Example chunk metadata: {document_chunks[0].metadata}")
                
            # Create the vector store
            vector_db = create_vector_store(document_chunks, VECTOR_DB_DIR)
            if vector_db:
                logger.info(f"Vector store successfully created at {VECTOR_DB_DIR}")
            else:
                logger.error("Failed to create vector store, cannot perform queries.")
                return
                
        # Test the vector store with a sample query
        # logger.info("Testing vector store with a sample query...")
        # test_queries = [
        #     "What are the symptoms of an elbow fracture?",
        #     "How are hand fractures treated?",
        #     "What's the recovery time for a shoulder fracture?"
        # ]
        
        # for query in test_queries:
        #     logger.info(f"Testing query: '{query}'")
        #     response = query_vector_store(vector_db, query)
        #     logger.info(f"Response preview: {response[:100]}...")
        #     print(f"\nQuery: {query}\nResponse: {response}\n{'-'*50}")
            
        # logger.info("Vector store test completed.")
        
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
        print(f"An error occurred: {e}. Please check the logs for details.")


if __name__ == "__main__":
    main()