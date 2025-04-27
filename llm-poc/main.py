# Module imports
from config import load_api_key, VECTOR_DB_DIR, TEMP_PDF_DIR, DATA_DIR
from pdf_processing import extract_zip, process_single_pdf
from text_processing import create_chunks_with_structured_data
from vector_store import create_vector_store, load_vector_store

# Library imports
import os
import shutil 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def load_and_process_pdfs(zip_path):
    """Processes a zipped folder of PDFs, preserves headers as metadata, and splits into smaller chunks for vector storage."""
    pdf_files = extract_zip(zip_path, extract_to=TEMP_PDF_DIR)
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for pdf in pdf_files:
        markdown_sections = process_single_pdf(pdf)
        enriched_chunks = create_chunks_with_structured_data(markdown_sections, text_splitter)
        for chunk in enriched_chunks:
            chunk.metadata["source"] = os.path.basename(pdf) # Ensure source is always present
            all_chunks.append(chunk)
    return all_chunks

def main():
    load_api_key()
    zip_file_path = DATA_DIR

    # --- Force recreation of the vector store ---
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Removing existing vector store at {VECTOR_DB_DIR} to force recreation...")
        shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
    # --- End of force recreation ---

    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print("Loading existing vector store...")
        vector_db = load_vector_store(VECTOR_DB_DIR)
    else:
        print("Processing PDFs and building the vector store...")
        document_chunks = load_and_process_pdfs(zip_file_path)
        print(f"Total chunks created: {len(document_chunks)}")

        # Tests
        if document_chunks:
            print(f"Example chunk:\n{document_chunks[0].page_content[:500]}")
            print(f"Metadata: {document_chunks[0].metadata}")

        vector_db = create_vector_store(document_chunks, VECTOR_DB_DIR)
        
        if vector_db:
            print(f"Vector store successfully created at {VECTOR_DB_DIR}")
        else:
            print("Failed to create vector store, cannot perform queries.")


if __name__ == "__main__":
    main()