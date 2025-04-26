# inference_functional.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import os

# Load models
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")

# Category mappings
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def predict_body_part(img_array):
    preds = model_parts.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_index]
    return categories_parts[predicted_index], confidence

def predict_fracture(img_array, body_part):
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
    confidence = preds[0][predicted_index]
    return categories_fracture[predicted_index], confidence

def inference(img_path, user_input):
    img_array = load_and_prepare_image(img_path)

    user_input_lower = user_input.lower()
    detected_parts = [part for part in ["elbow", "hand", "shoulder"] if part in user_input_lower]

    if len(detected_parts) == 1:
        body_part = detected_parts[0].capitalize()
        fracture_status, fracture_confidence = predict_fracture(img_array, body_part)
    else:
        body_part, part_confidence = predict_body_part(img_array)
        fracture_status, fracture_confidence = predict_fracture(img_array, body_part)

    result = {
        "body_part": body_part,
        "fracture_status": fracture_status,
        "confidence": fracture_confidence
    }
    return result


if __name__ == "__main__":
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    img_path = THIS_FOLDER + "/aaa867f71f0fedbd9cdadd08e62a17_gallery.jpeg"
    user_input = "I am in pain, help me analyze this x-ray image."

    result = inference(img_path, user_input)

    print(result)
