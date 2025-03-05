import torch
from torchvision import models, transforms
import qai_hub
from typing import Tuple


# Custom wrapper class for preprocessing and MobileNetV2
class PreprocessedMobileNetV2(torch.nn.Module):
    def __init__(self, num_classes, pretrained_weights_path):
        super(PreprocessedMobileNetV2, self).__init__()
        # Load MobileNetV2 with the specified number of classes
        self.mobilenet_v2 = models.mobilenet_v2(num_classes=num_classes)

        # Load pretrained weights from .pth file
        state_dict = torch.load(pretrained_weights_path, weights_only=True)
        self.mobilenet_v2.load_state_dict(state_dict)

        # Define preprocessing steps
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, img):
        # Apply preprocessing
        img_tensor = self.preprocess(img)

        # Pass the preprocessed image through the model
        return self.mobilenet_v2(img_tensor)


def run_inference(model, device, input_dataset):
    """Submit an inference job for the model."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
    )
    return inference_job.download_output_data()


# Parameters
num_classes = 64

pretrained_path = "./model/mobilenet_v2_coco.pth"  # Replace with your .pth file path

# Create the model
model = PreprocessedMobileNetV2(
    num_classes=num_classes, pretrained_weights_path=pretrained_path
)

# Inference
model.eval()

# Trace model
input_shape: Tuple[int, ...] = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
pt_model = torch.jit.trace(model, example_input)

# Compile model on a specific device
compile_job = qai_hub.submit_compile_job(
    pt_model,
    name="coco_imagenet",  # Replace with your model name
    device=qai_hub.Device("Samsung Galaxy S24 (Family)"),
    input_specs=dict(image=input_shape),
)

compile_job.modify_sharing(
    add_emails=["lowpowervision@gmail.com"]
)  ## Share your model for submission

compiled_model = compile_job.get_target_model()

# inference_output = run_inference(compiled_model, device = qai_hub.Device("Samsung Galaxy S24 (Family)"), input_dataset = qai_hub.get_dataset(""))
# output_array = inference_output['output_0']
