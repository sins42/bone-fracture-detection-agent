import nltk
nltk.download('punkt')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_chunks_with_structured_data(structured_data, text_splitter):
    enriched_chunks = []
    current_heading = None
    for item in structured_data:
        metadata = {}
        content_to_split = ""
        if item["type"] == "heading":
            current_heading = item["content"]
            continue # Don't chunk headings themselves, just use for context
        elif item["type"] == "paragraph":
            content_to_split = item["content"]
            if current_heading:
                metadata["heading"] = current_heading
        elif item["type"] == "list":
            content_to_split = "\n".join(item["items"])
            metadata["is_list"] = True
            if current_heading:
                metadata["heading"] = current_heading
        elif item["type"] == "page_content":
            content_to_split = item["content"]
            metadata["is_page_content"] = True

        if content_to_split:
            chunks = text_splitter.split_text(content_to_split)
            for chunk in chunks:
                enriched_chunks.append(Document(page_content=chunk, metadata=metadata))
    return enriched_chunks