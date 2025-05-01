# text_processing.py
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.docstore.document import Document
import logging
from typing import List, Dict, Any

# Initialize logger
logger = logging.getLogger(__name__)

def create_chunks_with_structured_data(markdown_sections: List[dict], text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Splits markdown sections into smaller, more semantically relevant chunks.
    
    Args:
        markdown_sections (List[dict]): List of sections extracted from the PDF
        text_splitter: The text splitter to use
        
    Returns:
        List[Document]: List of Document objects with metadata
    """
    enriched_chunks = []
    
    for section in markdown_sections:
        # Extract and sanitize data
        section_type = section.get("type", "paragraph")
        content = section.get("content", "").strip()
        source = section.get("source", "unknown")
        
        # Skip empty sections
        if not content:
            continue
            
        # Clean the text
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Process based on section type
        if section_type == "heading":
            # For headings, add them as-is with special metadata
            enriched_chunks.append(Document(
                page_content=content,
                metadata={
                    "section_type": "heading",
                    "source": source,
                    "is_heading": True
                }
            ))
        else:
            # For paragraphs, split by semantic units
            if len(content) <= CHUNK_SIZE:
                # Short content goes in as a single chunk
                enriched_chunks.append(Document(
                    page_content=content,
                    metadata={
                        "section_type": "paragraph",
                        "source": source,
                        "is_heading": False
                    }
                ))
            else:
                # Longer content needs splitting
                # First try to split by paragraphs
                paragraphs = re.split(r'\n\s*\n', content)
                
                if len(paragraphs) > 1:
                    # Process each paragraph
                    for i, para in enumerate(paragraphs):
                        if not para.strip():
                            continue
                            
                        if len(para) <= CHUNK_SIZE:
                            enriched_chunks.append(Document(
                                page_content=para.strip(),
                                metadata={
                                    "section_type": "paragraph",
                                    "paragraph_index": i,
                                    "source": source,
                                    "is_heading": False
                                }
                            ))
                        else:
                            # Use sentence splitting for long paragraphs
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            current_chunk = ""
                            
                            for sentence in sentences:
                                if len(current_chunk) + len(sentence) < CHUNK_SIZE:
                                    current_chunk += sentence + " "
                                else:
                                    if current_chunk:
                                        enriched_chunks.append(Document(
                                            page_content=current_chunk.strip(),
                                            metadata={
                                                "section_type": "paragraph",
                                                "paragraph_index": i,
                                                "source": source,
                                                "is_heading": False
                                            }
                                        ))
                                    current_chunk = sentence + " "
                                    
                            # Add the last chunk if it exists
                            if current_chunk:
                                enriched_chunks.append(Document(
                                    page_content=current_chunk.strip(),
                                    metadata={
                                        "section_type": "paragraph",
                                        "paragraph_index": i,
                                        "source": source,
                                        "is_heading": False
                                    }
                                ))
                else:
                    # Use the token-based text splitter for long single paragraphs
                    chunks = text_splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        enriched_chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                "section_type": "paragraph",
                                "chunk_index": i,
                                "source": source,
                                "is_heading": False
                            }
                        ))
    
    logger.info(f"Created {len(enriched_chunks)} chunks from {len(markdown_sections)} sections")
    return enriched_chunks

def process_single_pdf(file_path: str) -> List[dict]:
    """
    Processes a single PDF file, extracts the text, and splits it into sections.
    Enhanced to handle multiple PDF formats and extract structured data.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[dict]: A list of dictionaries representing structured sections
    """
    try:
        import pdfplumber
        
        structured_sections = []
        
        # Use pdfplumber for more reliable text extraction
        with pdfplumber.open(file_path) as pdf:
            # Process each page
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # Split the text into lines
                lines = text.split("\n")
                current_heading = None
                current_content = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line is a heading
                    # Pattern matches capitalized text or numbered sections
                    if re.match(r"^(?:[A-Z][A-Z\s]+:|[0-9]+\.\s+[A-Z])", line):
                        # Save the previous section if it exists
                        if current_content and current_heading is not None:
                            structured_sections.append({
                                "type": "paragraph",
                                "content": current_content.strip(),
                                "heading": current_heading,
                                "source": os.path.basename(file_path),
                                "page": page_num + 1
                            })
                            current_content = ""
                        
                        # Start a new section
                        current_heading = line
                        structured_sections.append({
                            "type": "heading",
                            "content": line,
                            "source": os.path.basename(file_path),
                            "page": page_num + 1
                        })
                    else:
                        # This is content text
                        current_content += line + " "
                
                # Add the last section if there's content
                if current_content:
                    structured_sections.append({
                        "type": "paragraph",
                        "content": current_content.strip(),
                        "heading": current_heading or "No Heading",
                        "source": os.path.basename(file_path),
                        "page": page_num + 1
                    })
        
        logger.info(f"Extracted {len(structured_sections)} sections from {file_path}")
        return structured_sections

    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        return []