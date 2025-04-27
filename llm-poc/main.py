# main.py
# Module imports
from config import load_api_key, VECTOR_DB_DIR, TEMP_PDF_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP # Import the chunking variables
from pdf_processing import extract_zip, process_single_pdf
from text_processing import create_chunks_with_structured_data
from vector_store import create_vector_store, load_vector_store
import logging
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_pdfs(zip_path):
    """
    Processes a zipped folder of PDFs, extracts text, creates chunks with structured
    data, and adds source file information to the metadata.

    Args:
        zip_path (str): Path to the zipped folder of PDFs.

    Returns:
        list: A list of Document objects, or an empty list on error.
    """
    try:
        pdf_files = extract_zip(zip_path, extract_to=TEMP_PDF_DIR)
        if not pdf_files:
            logging.warning("No PDF files found in the zip archive.")
            return []

        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) # Use the values from config.py

        for pdf in pdf_files:
            markdown_sections = process_single_pdf(pdf)
            if not markdown_sections:
                logging.warning(f"No markdown sections extracted from {pdf}. Skipping.")
                continue  # Skip to the next PDF

            enriched_chunks = create_chunks_with_structured_data(markdown_sections, text_splitter)
            for chunk in enriched_chunks:
                chunk.metadata["source"] = os.path.basename(pdf)  # Ensure source is always present
                all_chunks.append(chunk)
        return all_chunks
    except Exception as e:
        logging.error(f"Error processing PDFs: {e}")
        return []  # Return empty list on error

def main():
    """
    Main function to load PDFs, create/load a vector store, and handle querying.
    """
    load_api_key()
    zip_file_path = DATA_DIR

    # --- Force recreation of the vector store ---
    if os.path.exists(VECTOR_DB_DIR):
        logging.info(f"Removing existing vector store at {VECTOR_DB_DIR} to force recreation...")
        shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
    # --- End of force recreation ---

    try:
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            logging.info("Loading existing vector store...")
            vector_db = load_vector_store(VECTOR_DB_DIR)
        else:
            logging.info("Processing PDFs and building the vector store...")
            document_chunks = load_and_process_pdfs(zip_file_path)
            if not document_chunks:
                logging.error("No document chunks were generated. Cannot create vector store.")
                return  # Exit if no chunks

            logging.info(f"Total chunks created: {len(document_chunks)}")

            # Tests
            if document_chunks:
                logging.info(f"Example chunk:\n{document_chunks[0].page_content[:500]}")
                logging.info(f"Metadata: {document_chunks[0].metadata}")

            vector_db = create_vector_store(document_chunks, VECTOR_DB_DIR)
            if vector_db:
                logging.info(f"Vector store successfully created at {VECTOR_DB_DIR}")
            else:
                logging.error("Failed to create vector store, cannot perform queries.")
                return  # Exit if vector store creation fails

        # Example query (for testing)
        # query = "What are the symptoms of an elbow fracture?"
        # response = query_vector_store(vector_db, query)
        # print(f"Response to query: '{query}':\n{response}")

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
        print(f"An error occurred: {e}.  Please check the logs for details.") # Inform the user.

if __name__ == "__main__":
    main()