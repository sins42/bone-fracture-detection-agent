import os
import zipfile
import pdfplumber
import re

def extract_zip(uploaded_zip_path, extract_to="../temp_pdfs"):
    """Extracts a zipped folder containing PDFs."""
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(uploaded_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    pdf_files = []
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    return pdf_files

def pdf_to_markdown_string(pdf_path):
    """
    Extracts text from a PDF and attempts to identify headings and paragraphs.
    """
    structured_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            current_block = ""
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    if current_block:
                        structured_content.append({"type": "paragraph", "content": current_block})
                        current_block = ""
                    continue

                # Basic heading detection (adjust regex as needed for your PDFs)
                if re.match(r"^(?:[A-Z]+(?: [A-Z]+)*|[0-9]+\.\s)", stripped_line):
                    if current_block:
                        structured_content.append({"type": "paragraph", "content": current_block})
                        current_block = ""
                    structured_content.append({"type": "heading", "content": stripped_line})
                else:
                    current_block += (stripped_line + " ") # Join words in a paragraph

            if current_block:
                structured_content.append({"type": "paragraph", "content": current_block.strip()})

    return structured_content

def process_single_pdf(pdf_path):
    """Processes a single PDF to extract structured content with tags."""
    structured_content = pdf_to_markdown_string(pdf_path)
    return structured_content