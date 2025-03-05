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
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Create the model
model = PreprocessedMobileNetV2(
    num_classes=num_classes, pretrained_weights_path=pretrained_path
)

# Inference
model.eval()

# Trace model
input_shape: Tuple[int, ...] = (1, 3, 224, 224)
img = Image.open("./data/handbag_indoor_lowlight_02.jpg")

with torch.inference_mode():
    output = model(img)
    probs = F.softmax(output, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    print(COCO_LPCV_CLASSES[pred])
