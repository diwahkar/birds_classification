import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os

from bird_model import BirdsClassifier
from constants import IMG_SIZE, NUM_CLASSES, LABELS, MODEL_SAVE_PATH


def predict_image(image_path, model_path=MODEL_SAVE_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BirdsClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    try:
        img_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            raise ValueError(f"Could not read image {image_path}.")

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]


        pil_img = Image.fromarray(img_rgb)

        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0)

    except Exception as e:
        raise RuntimeError(f"Error processing image {image_path}: {e}")

    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)

    probabilities = F.softmax(output, dim=1)

    predicted_prob, predicted_idx = torch.max(probabilities, 1)

    predicted_label = LABELS[predicted_idx.item()]
    confidence = predicted_prob.item()

    all_probabilities_dict = {LABELS[i]: prob.item() for i, prob in enumerate(probabilities[0])}

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "all_probabilities": all_probabilities_dict
    }

if __name__ == "__main__":
    default_image_path = 'bird_1.png'
    image_to_predict = input(f"Enter the path to the bird image to predict (e.g., 'BirdImages/b0/some_image.jpg' or press Enter to use '{default_image_path}'): ")

    if not image_to_predict:
        image_to_predict = default_image_path

    try:
        prediction_results = predict_image(image_to_predict)
        print(f"\nPrediction for {image_to_predict}:")
        print(f"Predicted Bird Class: {prediction_results['predicted_class']}")
        print(f"Confidence: {prediction_results['confidence']:.4f}")
        print("\nAll Class Probabilities:")
        for label, prob in prediction_results['all_probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Prediction failed: {e}")
