# prediction_for_chatbot.py
import numpy as np
import tensorflow as tf
import tf_keras as tf_keras
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Model paths - make these configurable
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), "Bone-Fracture-Detection", "weights")
MODEL_ELBOW = os.path.join(MODEL_DIR, "ResNet50_Elbow_frac.h5")
MODEL_HAND = os.path.join(MODEL_DIR, "ResNet50_Hand_frac.h5")
MODEL_SHOULDER = os.path.join(MODEL_DIR, "ResNet50_Shoulder_frac.h5")
MODEL_PARTS = os.path.join(MODEL_DIR, "ResNet50_BodyParts.h5")

# Initialize models to None - we'll load them on demand
model_elbow_frac = None
model_hand_frac = None
model_shoulder_frac = None
model_parts = None

# Category mappings
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def load_models():
    """
    Load all necessary models on first use
    """
    global model_elbow_frac, model_hand_frac, model_shoulder_frac, model_parts
    
    try:
        logger.info("Loading classification models...")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(MODEL_DIR):
            logger.error(f"Model directory not found: {MODEL_DIR}")
            return False
            
        # Check if model files exist
        if not os.path.exists(MODEL_PARTS):
            logger.error(f"Body parts model not found: {MODEL_PARTS}")
            return False
            
        # Load models
        model_parts = tf.keras.models.load_model(MODEL_PARTS)
        model_elbow_frac = tf.keras.models.load_model(MODEL_ELBOW)
        model_hand_frac = tf.keras.models.load_model(MODEL_HAND)
        model_shoulder_frac = tf.keras.models.load_model(MODEL_SHOULDER)
        
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def load_and_prepare_image(img_path):
    """
    Loads and preprocesses an image for model prediction
    """
    try:
        img = tf_keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf_keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def predict_body_part(img_array):
    """
    Predicts which body part is shown in the X-ray image
    """
    global model_parts
    
    # Load models if not already loaded
    if model_parts is None:
        if not load_models():
            raise ValueError("Failed to load classification models")
    
    preds = model_parts.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][predicted_index])  # Convert to Python float for JSON serialization
    return categories_parts[predicted_index], confidence

def predict_fracture(img_array, body_part):
    """
    Predicts whether there is a fracture in the specified body part
    """
    global model_elbow_frac, model_hand_frac, model_shoulder_frac
    
    # Load models if not already loaded
    if model_elbow_frac is None or model_hand_frac is None or model_shoulder_frac is None:
        if not load_models():
            raise ValueError("Failed to load classification models")
    
    if body_part == "Elbow":
        model = model_elbow_frac
    elif body_part == "Hand":
        model = model_hand_frac
    elif body_part == "Shoulder":
        model = model_shoulder_frac
    else:
        raise ValueError(f"Unknown body part: {body_part}")

    preds = model.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][predicted_index])  # Convert to Python float for JSON serialization
    return categories_fracture[predicted_index], confidence

def inference(img_path, user_input):
    """
    Main inference function that processes an X-ray image and returns analysis results
    
    Args:
        img_path (str): Path to the X-ray image
        user_input (str): The user's query or description
        
    Returns:
        dict: Dictionary containing classification results
    """
    try:
        logger.info(f"Running inference on image: {img_path}")
        img_array = load_and_prepare_image(img_path)

        # Extract body part from user input if specified
        user_input_lower = user_input.lower()
        detected_parts = [part for part in ["elbow", "hand", "shoulder"] if part in user_input_lower]

        if len(detected_parts) == 1:
            # User specified which body part
            body_part = detected_parts[0].capitalize()
            logger.info(f"User specified body part: {body_part}")
            fracture_status, fracture_confidence = predict_fracture(img_array, body_part)
            # We still predict the body part to double-check
            predicted_part, part_confidence = predict_body_part(img_array)
            
            # If predictions don't match, include a note
            part_match = predicted_part == body_part
            
        else:
            # Predict body part from image
            body_part, part_confidence = predict_body_part(img_array)
            logger.info(f"Predicted body part: {body_part}")
            fracture_status, fracture_confidence = predict_fracture(img_array, body_part)
            part_match = True

        result = {
            "body_part": body_part,
            "fracture_status": fracture_status,
            "confidence": fracture_confidence,
            "part_confidence": part_confidence,
            "part_match": part_match
        }
        
        logger.info(f"Inference result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise

if __name__ == "__main__":
    # Setup basic logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(THIS_FOLDER, "test_image.jpg")
    user_input = "I am in pain, help me analyze this x-ray image."

    result = inference(img_path, user_input)
    print(result)