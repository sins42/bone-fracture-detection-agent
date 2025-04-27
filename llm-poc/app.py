# app.py
import gradio as gr
import logging
from config import load_api_key, VECTOR_DB_DIR
from vector_store import load_vector_store, query_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API key and vector store on startup
try:
    load_api_key()
    vector_db = load_vector_store(VECTOR_DB_DIR)
    if vector_db is None:
        logging.error("Failed to load vector store on startup.  The application may not function correctly.")
except Exception as e:
    logging.critical(f"Exception during startup: {e}")
    # Consider if you want to exit here, or show a degraded UI.
    # raise  # Re-raise if you want the app to crash on startup error.


def ask_skeletaX(user_query):
    """
    Function to handle user queries and return the agent's response.

    Args:
        user_query (str): The user's question.

    Returns:
        str: The agent's response, or an error message.
    """
    logging.info(f"User query: {user_query}")
    if vector_db:
        try:
            response = query_vector_store(vector_db, user_query)
            # Improved response parsing (assuming LLM returns JSON)
            try:
                import json
                response_json = json.loads(response)
                answer = response_json.get("answer", response) # Default to the whole response if 'answer' not found
                thought = response_json.get("thought", "")
                if thought:
                    logging.info(f"LLM Thought: {thought}") # Log the LLM's thought process
                return answer
            except json.JSONDecodeError:
                logging.warning("Expected JSON response from LLM, but received: %s", response)
                return response # Return the raw response.

        except Exception as e:
            error_message = f"Error querying vector store: {e}"
            logging.error(error_message)
            return f"Sorry, there was an error processing your query: {e}"
    else:
        error_message = "Error: Vector store not loaded. Please ensure the backend is set up correctly."
        logging.error(error_message)
        return error_message



if __name__ == "__main__":
    iface = gr.Interface(
        fn=ask_skeletaX,
        inputs=[
            gr.Textbox(label="Ask SkeletaX a question about Bone Fractures:"),
        ],
        outputs=gr.Textbox(label="SkeletaX's Response:"),
        title="SkeletaX: The Future of Fracture Care",
        description="Got bone fracture questions? Ask SkeletaX for immediate insights.",
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)  # Make it accessible on your network
