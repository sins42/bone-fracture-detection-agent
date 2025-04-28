# pdf_processing.py
import os
import zipfile
import logging
from typing import List, Dict, Any, Optional
import re

# Set up logging
logger = logging.getLogger(__name__)

def extract_zip(uploaded_zip_path: str, extract_to: str = "../temp_pdfs") -> List[str]:
    """
    Extracts a zipped folder containing PDFs.
    
    Args:
        uploaded_zip_path (str): Path to the zip file
        extract_to (str): Directory to extract files to
        
    Returns:
        List[str]: List of paths to extracted PDF files
    """
    try:
        # Create the extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        
        # Check if the zip file exists
        if not os.path.exists(uploaded_zip_path):
            logger.error(f"Zip file not found: {uploaded_zip_path}")
            return []
            
        logger.info(f"Extracting {uploaded_zip_path} to {extract_to}")
        
        with zipfile.ZipFile(uploaded_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Find all extracted PDF files
        pdf_files = []
        for root, _, files in os.walk(extract_to):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    pdf_files.append(pdf_path)
                    logger.info(f"Found PDF: {pdf_path}")
        
        logger.info(f"Extracted {len(pdf_files)} PDF files")
        return pdf_files
        
    except zipfile.BadZipFile:
        logger.error(f"The file is not a valid zip file: {uploaded_zip_path}")
        return []
    except Exception as e:
        logger.error(f"Error extracting zip file: {e}")
        return []

def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF.
    
    Args:
        text (str): Raw text from PDF
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = text.replace('- ', '')  # Remove hyphenation
    
    # Fix missing spaces after periods
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    return text.strip()

def process_single_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Main function to process a PDF file. This is the only function that should be
    called externally, and it imports the appropriate PDF library as needed.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, Any]]: Structured content from the PDF
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        import pdfplumber
        
        # Make sure the file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
            
        structured_content = []
        source_filename = os.path.basename(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            current_section = {"type": "paragraph", "content": "", "source": source_filename}
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                    
                text = clean_text(text)
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if this is a heading (all caps or numbered)
                    if re.match(r'^[A-Z0-9\s\-:.,]{3,}$', line) or re.match(r'^[0-9]+\.\s+', line):
                        # Save current section if not empty
                        if current_section["content"]:
                            structured_content.append(current_section)
                            
                        # Start a new section with this heading
                        structured_content.append({
                            "type": "heading",
                            "content": line,
                            "source": source_filename,
                            "page": page_num + 1
                        })
                        
                        # Reset current section
                        current_section = {
                            "type": "paragraph", 
                            "content": "", 
                            "source": source_filename,
                            "page": page_num + 1
                        }
                    else:
                        # This is paragraph content - append to current section
                        if current_section["content"]:
                            current_section["content"] += " " + line
                        else:
                            current_section["content"] = line
                
            # Add the last section if not empty
            if current_section["content"]:
                structured_content.append(current_section)
                
        logger.info(f"Extracted {len(structured_content)} sections from {pdf_path}")
        return structured_content
        
    except ImportError:
        logger.error("pdfplumber library not found. Please install it with 'pip install pdfplumber'")
        return []
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return []