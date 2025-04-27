# text_processing.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import logging
import re  # Import the regular expression module

# Initialize logger
logger = logging.getLogger(__name__)

def create_chunks_with_structured_data(markdown_sections, text_splitter):
    """
    Splits markdown sections into smaller chunks, preserving structure and adding metadata.

    Args:
        markdown_sections (list): A list of dictionaries, where each dictionary
            represents a markdown section with keys like "heading" and "content".
        text_splitter (TextSplitter): A text splitter object (e.g., RecursiveCharacterTextSplitter).

    Returns:
        list: A list of Document objects, where each Document represents a text chunk
            with the original markdown section information added to the metadata.
    """
    enriched_chunks = []
    for section in markdown_sections:
        heading = section.get("heading", "No Heading")  # Default to "No Heading"
        content = section.get("content", "")
        is_list = section.get("is_list", False)

        # Clean the text to remove unnecessary whitespace and newlines
        content = re.sub(r'\s+', ' ', content).strip()

        if not content:
            logger.warning(f"Skipping empty content for heading: {heading}")
            continue

        # Split the text into smaller chunks
        chunks = text_splitter.create_documents([content])  # Pass content as a list

        for i, chunk in enumerate(chunks):
            # Add the original markdown section information to the metadata
            chunk.metadata["heading"] = heading
            chunk.metadata["is_list"] = is_list
            chunk.metadata["chunk_id"] = f"{heading}-{i}"  # Unique chunk identifier
            enriched_chunks.append(chunk)
    return enriched_chunks


def process_single_pdf(file_path):
    """
    Processes a single PDF file, extracts the text, and splits it into markdown-like sections.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a markdown section
            with keys like "heading" and "content".  Returns an empty list on error.
    """
    try:
        # Import the necessary library here to avoid a top-level import error.
        import PyPDF2
        from io import BytesIO

        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # changed from extractText to extract_text

        # Basic markdown-like section extraction (this will need improvement)
        sections = []
        # Improved regex for headings (handles multiple levels and formatting)
        heading_regex = r"(#+)(.*?)\n"
        last_match_end = 0

        for match in re.finditer(heading_regex, text):
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            start_index = match.start()
            content = text[last_match_end:start_index].strip()
            sections.append({"heading": heading_text, "content": content, "is_list": False})
            last_match_end = match.end()

        # Add the last section (after the last heading or if no headings)
        if last_match_end < len(text):
            sections.append({"heading": "No Heading", "content": text[last_match_end:].strip(), "is_list": False})

        return sections

    except Exception as e:
        logger.error(f"Error processing PDF file: {file_path} - {e}")
        return []  # Return an empty list on error