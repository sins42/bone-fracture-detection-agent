import os
from colorama import Fore
from predictions import predict


def load_path(path):
    dataset = []
    for body in os.listdir(path):
        path_p = os.path.join(path, body)
        for lab in os.listdir(path_p):
            path_l = os.path.join(path_p, lab)
            for img in os.listdir(path_l):
                dataset.append({
                    'body_part': body,
                    'label': lab,
                    'image_path': os.path.join(path_l, img),
                    'image_name': img
                })
    return dataset


def evaluate_body_part_model(dataset):
    print(Fore.CYAN + "\n--- Evaluating Body Part Model ---")
    total = len(dataset)
    correct = 0

    for img in dataset:
        predicted = predict(img['image_path'], model="Parts")
        if predicted == img['body_part']:
            correct += 1

    acc = correct / total * 100
    print(Fore.BLUE + f"Body Part Classification Accuracy: {acc:.2f}%")
    return acc


def evaluate_fracture_model(dataset, body_part):
    print(Fore.CYAN + f"\n--- Evaluating Fracture Model for {body_part} ---")
    filtered = [img for img in dataset if img['body_part'] == body_part]
    if not filtered:
        print(Fore.YELLOW + f"No samples for {body_part}.")
        return None

    correct = 0
    for img in filtered:
        predicted = predict(img['image_path'], model=body_part)
        if predicted == img['label']:
            correct += 1

    acc = correct / len(filtered) * 100
    print(Fore.BLUE + f"{body_part} Fracture Classification Accuracy: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(THIS_FOLDER, 'test')
    dataset = load_path(test_dir)

    # Evaluate each model
    evaluate_body_part_model(dataset)
    for part in ["Elbow", "Hand", "Shoulder"]:
        evaluate_fracture_model(dataset, part)
