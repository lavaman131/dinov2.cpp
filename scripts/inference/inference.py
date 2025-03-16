from pathlib import Path
import torch
from torchvision import models
import torch.nn as nn
import qai_hub
from torchvision.transforms import v2
from typing import Tuple
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import torch.nn.functional as F
from efficient_cv.data import COCO_LPCV_CLASSES
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm.auto import tqdm

from efficient_cv.evaluate.metrics import calculate_accuracy


# Custom wrapper class for preprocessing and MobileNetV2
class PreprocessedMobileNetV2(nn.Module):
    def __init__(self, num_classes, pretrained_weights_path):
        super(PreprocessedMobileNetV2, self).__init__()
        # Load MobileNetV2 with the specified number of classes
        self.mobilenet_v2 = mobilenet_v2(num_classes=num_classes)

        # Load pretrained weights from .pth file
        state_dict = torch.load(pretrained_weights_path, weights_only=True)
        self.mobilenet_v2.load_state_dict(state_dict)

        # Define preprocessing steps
        self.preprocess = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def forward(self, img):
        # Apply preprocessing
        img_tensor = self.preprocess(img).unsqueeze_(0)

        # Pass the preprocessed image through the model
        return self.mobilenet_v2(img_tensor)


# Parameters
num_classes = 64

pretrained_path = "./models/mobilenet_v2_coco.pth"  # Replace with your .pth file path
ground_truth_path = "./data/labels/key.csv"

# Create the model
model = PreprocessedMobileNetV2(
    num_classes=num_classes, pretrained_weights_path=pretrained_path
)

# Inference
model.eval()


outputs = []
fnames = []

image_paths = list(Path("./data/images").iterdir())

with torch.inference_mode():
    for img_path in tqdm(image_paths):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            output = model(img)
            fnames.append(img_path.name)
        outputs.append(output)

outputs = torch.cat(outputs, dim=0)

accuracy = calculate_accuracy(
    outputs=outputs, file_names=fnames, ground_truth_path=ground_truth_path
)

print(f"Accuracy: {accuracy}")
