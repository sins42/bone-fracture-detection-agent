import gradio as gr
import os
from config import load_api_key, VECTOR_DB_DIR
from vector_store import load_vector_store, query_vector_store

# Load API key and vector store on startup
load_api_key()
vector_db = load_vector_store(VECTOR_DB_DIR)

def ask_skeletaX(user_query):
    """
    Function to handle user queries and return the agent's response.
    """
    if vector_db:
        response = query_vector_store(vector_db, user_query)
        if "<think>" in response and "</think>" in response:
            start_index = response.find("</think>") + len("</think>")
            return response[start_index:].strip()
        return response
    else:
        return "Error: Vector store not loaded. Please ensure the backend is set up correctly."

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
    iface.launch(server_name="0.0.0.0", server_port=7860) # Make it accessible on your network