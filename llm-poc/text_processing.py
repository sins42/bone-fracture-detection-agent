# text_processing.py
from config import (
    load_api_key,
    VECTOR_DB_DIR,
    TEMP_PDF_DIR,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)  # Explicitly import variables
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.docstore.document import Document
import logging
import re

# Initialize logger
logger = logging.getLogger(__name__)

def create_chunks_with_structured_data(markdown_sections, text_splitter):
    """
    Splits markdown sections into smaller, more semantically relevant chunks.
    """
    enriched_chunks = []
    for section in markdown_sections:
        heading = section.get("heading", "No Heading")
        content = section.get("content", "")
        is_list = section.get("is_list", False)

        # Clean the text more aggressively
        content = re.sub(r'\s+', ' ', content).strip()
        content = re.sub(r'[^a-zA-Z0-9\s.,;!?(){}\[\]\'\"]', '', content)

        if not content:
            logger.warning(f"Skipping empty content for heading: {heading}")
            continue

        # Use a Sentence Splitter
        # chunks = text_splitter.create_documents([content])  # Old way
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            enriched_chunks.append(Document(page_content=chunk, metadata={"heading": heading, "is_list": is_list, "chunk_id": f"{heading}-{i}"}))
    return enriched_chunks


def process_single_pdf(file_path):
    """
    Processes a single PDF file, extracts the text, and splits it into markdown-like sections.
    """
    try:
        import PyPDF2
        from io import BytesIO

        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

        sections = []
        heading_regex = r"(#+)(.*?)\n"
        last_match_end = 0

        for match in re.finditer(heading_regex, text):
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            start_index = match.start()
            content = text[last_match_end:start_index].strip()
            sections.append({"heading": heading_text, "content": content, "is_list": False})
            last_match_end = match.end()

        if last_match_end < len(text):
            sections.append({"heading": "No Heading", "content": text[last_match_end:].strip(), "is_list": False})

        return sections

    except Exception as e:
        logger.error(f"Error processing PDF file: {file_path} - {e}")
        return []
