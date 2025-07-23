import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import cv2

from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

os.makedirs("gradcam_results", exist_ok=True)

image_urls = {
    "West_Highland_white_terrier": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02098286_West_Highland_white_terrier.JPEG",
    "American_coot": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02018207_American_coot.JPEG",
    "racer": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n04037443_racer.JPEG",
    "flamingo": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02007558_flamingo.JPEG",
    "kite": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01608432_kite.JPEG",
    "goldfish": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01443537_goldfish.JPEG",
    "tiger_shark": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01491361_tiger_shark.JPEG",
    "vulture": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01616318_vulture.JPEG",
    "common_iguana": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01677366_common_iguana.JPEG",
    "orange": "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n07747607_orange.JPEG",
}

import requests

def download_image(url, filename):
    if not os.path.exists(filename):
        img_data = requests.get(url).content
        with open(filename, 'wb') as handler:
            handler.write(img_data)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    rgb_img = np.array(img) / 255.0
    rgb_img = np.float32(rgb_img)
    input_tensor = transform(img).unsqueeze(0)
    return rgb_img, input_tensor

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

target_layers = [model.layer4[-1]]

for name, url in image_urls.items():
    print(f"Processing {name}...")
    img_path = f"gradcam_results/{name}.jpg"
    download_image(url, img_path)

    rgb_img, input_tensor = preprocess_image(img_path)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    rgb_img_resized = cv2.resize(rgb_img, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
    gradcam_result = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
    Image.fromarray(gradcam_result).save(f"gradcam_results/{name}_GradCAM.jpg")

    ablation_cam = AblationCAM(model=model, target_layers=target_layers)
    ablation_map = ablation_cam(input_tensor=input_tensor)[0, :]
    ablation_result = show_cam_on_image(rgb_img_resized, ablation_map, use_rgb=True)
    Image.fromarray(ablation_result).save(f"gradcam_results/{name}_AblationCAM.jpg")

    score_cam = ScoreCAM(model=model, target_layers=target_layers)
    score_map = score_cam(input_tensor=input_tensor)[0, :]
    scorecam_result = show_cam_on_image(rgb_img_resized, score_map, use_rgb=True)
    Image.fromarray(scorecam_result).save(f"gradcam_results/{name}_ScoreCAM.jpg")

print("All Grad-CAM results have been saved in 'gradcam_results' folder.")
