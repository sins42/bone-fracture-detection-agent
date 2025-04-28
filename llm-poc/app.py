# app.py
import gradio as gr
import logging
import os
import sys
import tempfile
from config import VECTOR_DB_DIR
from vector_store import load_vector_store, query_vector_store

# Get the absolute path to the directory containing Bone-Fracture-Detection
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add it to the Python path
sys.path.append(parent_dir)

# Use importlib to handle the import with hyphens
import importlib
module = importlib.import_module("Bone-Fracture-Detection.prediction_for_chatbot")
inference = module.inference

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


def process_xray_and_query(image, user_query):
    """
    Function to process uploaded X-ray image and user query.
    
    Args:
        image: The uploaded X-ray image
        user_query (str): The user's question
        
    Returns:
        str: The detailed analysis response
    """
    logger.info(f"Processing X-ray image and query: {user_query}")
    
    # Check if query is empty
    if not user_query or not user_query.strip():
        return "Please enter a question along with your X-ray image."
    
    # Check if image was uploaded
    if image is None:
        return "Please upload an X-ray image for analysis."
    
    # Save uploaded image to a temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
            temp_img_path = temp_img.name
            image.save(temp_img_path)
            logger.info(f"Saved uploaded image to {temp_img_path}")
        
        # Process the image with the classifier
        classification_result = inference(temp_img_path, user_query)
        
        # Clean up the temporary file
        os.unlink(temp_img_path)
        
        # Generate text response based on the user query
        global vector_db
        if vector_db is None:
            if not initialize_vector_store():
                return "The knowledge base is currently unavailable. Please try again later."
        
        # Enhance the query with the classification results
        enhanced_query = f"Information about {classification_result['body_part']} fracture: {user_query}"
        text_response = query_vector_store(vector_db, enhanced_query)
        
        # Format the detailed report
        confidence_percentage = f"{classification_result['confidence'] * 100:.1f}%"
        
        report = f"""# X-Ray Analysis Report

## Classification Results:
- **Body Part**: {classification_result['body_part']}
- **Diagnosis**: {classification_result['fracture_status'].capitalize()} 
- **Confidence**: {confidence_percentage}

## Expert Analysis:
{text_response}

*Note: This is an AI-assisted analysis and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.*
"""
        
        logger.info(f"Generated X-ray analysis report successfully")
        return report
        
    except Exception as e:
        error_message = f"Error processing X-ray image: {e}"
        logger.error(error_message)
        return f"I encountered a problem processing your X-ray image: {str(e)}"


def main():
    """Main function to run the Gradio interface."""
    # Initialize the vector store
    if not initialize_vector_store():
        logger.warning("Running with uninitialized vector store. Some queries may fail.")
    
    # Create examples for the text interface
    text_examples = [
        "What are the symptoms of an elbow fracture?",
        "How are hand fractures diagnosed?",
        "What treatments are available for shoulder fractures?",
        "How long does it take to recover from a wrist fracture?",
        "What exercises help rehabilitation after a bone fracture?"
    ]
    
    # Create the Gradio interface with two tabs
    with gr.Blocks(theme=gr.themes.Soft(), title="SkeletaX: The Future of Fracture Care") as demo:
        gr.Markdown("""
        # SkeletaX: The Future of Fracture Care
        
        Welcome to SkeletaX! I'm your bone health expert for elbow, hand, and shoulder fractures. 
        Ask me about symptoms, diagnosis, treatment options, recovery expectations, and best practices.
        """)
        
        with gr.Tabs():
            with gr.Tab("Text Chat"):
                text_input = gr.Textbox(
                    label="Ask SkeletaX a question about elbow, hand, or shoulder fractures:",
                    placeholder="Example: What are the common symptoms of a radial head fracture?",
                    lines=2
                )
                text_output = gr.Textbox(label="SkeletaX's Response:", lines=10)
                text_button = gr.Button("Ask Question")
                gr.Examples(text_examples, inputs=text_input)
                text_button.click(fn=ask_skeletaX, inputs=text_input, outputs=text_output)
                text_input.submit(fn=ask_skeletaX, inputs=text_input, outputs=text_output)
            
            with gr.Tab("X-Ray Analysis"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload X-Ray Image")
                    
                image_query = gr.Textbox(
                    label="What would you like to know about this X-ray?",
                    placeholder="Example: Can you tell me if there's a fracture and what treatment might be needed?",
                    lines=2
                )
                
                image_output = gr.Markdown(label="Analysis Report")
                image_button = gr.Button("Analyze X-Ray")
                
                image_examples = [
                    ["What can you tell me about this X-ray?"],
                    ["Is this a severe fracture? What treatment options do I have?"],
                    ["How long might recovery take for this injury?"]
                ]
                
                gr.Examples(image_examples, inputs=image_query)
                image_button.click(fn=process_xray_and_query, inputs=[image_input, image_query], outputs=image_output)
    
    # Launch the interface
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()