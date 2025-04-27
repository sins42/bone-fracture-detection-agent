import os
from PIL import Image
from predictions import predict

# Set image path directly
# Replace with your desired image path
image_path = "test/Elbow/fractured/elbow2.jpeg"

def main():
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Predict bone type
    bone_type = predict(image_path)
    print(f"Bone Type: {bone_type}")

    # Predict fracture result
    result = predict(image_path, bone_type)
    print(f"Fracture Detection Result: {result}")

    if result == 'fractured':
        print("Diagnosis: FRACTURED")
    else:
        print("Diagnosis: NORMAL")

if __name__ == "__main__":
    main()
