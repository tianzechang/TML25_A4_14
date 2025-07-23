#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

image_paths = {
    'West_Highland_white_terrier': 'gradcam_results/West_Highland_white_terrier.jpg',
    'American_coot'             : 'gradcam_results/American_coot.jpg',
    'racer'                      : 'gradcam_results/racer.jpg',
    'flamingo'                   : 'gradcam_results/flamingo.jpg',
    'kite'                       : 'gradcam_results/kite.jpg',
    'goldfish'                   : 'gradcam_results/goldfish.jpg',
    'tiger_shark'                : 'gradcam_results/tiger_shark.jpg',
    'vulture'                    : 'gradcam_results/vulture.jpg',
    'common_iguana'              : 'gradcam_results/common_iguana.jpg',
    'orange'                     : 'gradcam_results/orange.jpg'
}

lime_common_params = dict(
    top_labels   = 5,
    hide_color   = 0,
    num_samples  = 1000,
)

out_dir = Path("lime_results")
out_dir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "mps"
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet50_Weights.IMAGENET1K_V2
model  = resnet50(weights=weights).to(device).eval()

preprocess = weights.transforms()

def classifier_fn(images_np: list[np.ndarray]) -> np.ndarray:
    imgs = [preprocess(Image.fromarray(img)) for img in images_np]
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

explainer = lime_image.LimeImageExplainer()

lime_params_dict = {}

for name, path in image_paths.items():
    print(f"Processing {name} …")
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=classifier_fn,
        **lime_common_params
    )

    temp_img, mask = explanation.get_image_and_mask(
        label           = explanation.top_labels[0],
        positive_only   = True,
        num_features    = 5,
        hide_rest       = False
    )

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(mark_boundaries(temp_img, mask))
    save_path = out_dir / f"{name}_lime.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    lime_params_dict[name] = lime_common_params.copy()

with open("lime_params.pkl", "wb") as f:
    pickle.dump(lime_params_dict, f)

print("\n✅ LIME generation done!")
print(f"• Visualizations saved in: {out_dir.resolve()}")
print("• Parameters saved as: lime_params.pkl")
