# app.py
import gradio as gr
import logging
import os
import sys
from config import VECTOR_DB_DIR
from vector_store import load_vector_store, query_vector_store

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("skeletax_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global variable for the vector store
vector_db = None

def initialize_vector_store():
    """Initialize the vector store on startup."""
    global vector_db
    
    try:
        logger.info(f"Loading vector store from {VECTOR_DB_DIR}")
        
        if not os.path.exists(VECTOR_DB_DIR):
            logger.error(f"Vector store directory not found: {VECTOR_DB_DIR}")
            return False
            
        vector_db = load_vector_store(VECTOR_DB_DIR)
        
        if vector_db is None:
            logger.error("Failed to load vector store. The application may not function correctly.")
            return False
            
        logger.info("Vector store loaded successfully")
        return True
    except Exception as e:
        logger.critical(f"Exception during vector store initialization: {e}")
        return False


def ask_skeletaX(user_query):
    """
    Function to handle user queries and return the agent's response.

    Args:
        user_query (str): The user's question.

    Returns:
        str: The agent's response, or an error message.
    """
    logger.info(f"Received user query: {user_query}")
    
    # Check if query is empty
    if not user_query or not user_query.strip():
        return "Please enter a question about elbow, hand, or shoulder fractures."
    
    # Check if vector store is loaded
    global vector_db
    if vector_db is None:
        # Try to load vector store again
        if not initialize_vector_store():
            return "The knowledge base is currently unavailable. Please try again later."
    
    # Process the query
    try:
        response = query_vector_store(vector_db, user_query)
        
        # Check if we got a valid response
        if not response or response.lower().startswith("sorry") or "can't answer" in response.lower():
            logger.warning(f"No relevant information found for query: {user_query}")
            return ("I don't have enough information to answer that specific question. "
                   "I can help with questions about symptoms, diagnosis, treatment, "
                   "and recovery for elbow, hand, or shoulder fractures. "
                   "Could you try rephrasing your question?")
        
        logger.info(f"Generated response successfully")
        return response
        
    except Exception as e:
        error_message = f"Error processing query: {e}"
        logger.error(error_message)
        return "I encountered a problem while processing your question. Please try again or rephrase your question about bone fractures."


def main():
    """Main function to run the Gradio interface."""
    # Initialize the vector store
    if not initialize_vector_store():
        logger.warning("Running with uninitialized vector store. Some queries may fail.")
    
    # Create examples for the interface
    examples = [
        "What are the symptoms of an elbow fracture?",
        "How are hand fractures diagnosed?",
        "What treatments are available for shoulder fractures?",
        "How long does it take to recover from a wrist fracture?",
        "What exercises help rehabilitation after a bone fracture?"
    ]
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=ask_skeletaX,
        inputs=[
            gr.Textbox(
                label="Ask SkeletaX a question about elbow, hand, or shoulder fractures:",
                placeholder="Example: What are the common symptoms of a radial head fracture?",
                lines=2
            ),
        ],
        outputs=gr.Textbox(label="SkeletaX's Response:", lines=10),
        title="SkeletaX: The Future of Fracture Care",
        description="""
        Welcome to SkeletaX! I'm your bone health expert for elbow, hand, and shoulder fractures. 
        Ask me about symptoms, diagnosis, treatment options, recovery expectations, and best practices.
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    # Launch the interface
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()