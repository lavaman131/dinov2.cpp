import os
import numpy as np
from PIL import Image
import qai_hub


def process_image(image_path, input_shape):
    """Load and process an image from a local path to the required input shape."""
    image = (
        Image.open(image_path).convert("RGB").resize((input_shape[2], input_shape[3]))
    )
    input_array = np.expand_dims(
        np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
    )
    return input_array


# Sample folder path in Google Drive
sample_folder_path = ""  # fill in your image path

# Get a list of all image files and sort them by name (to maintain order)
sample_image_paths = sorted(
    [
        os.path.join(sample_folder_path, filename)
        for filename in os.listdir(sample_folder_path)
        if filename.endswith((".jpg", ".png", ".jpeg"))
    ]
)

# Specify the input shape
input_shape = (len(sample_image_paths), 3, 224, 224)

# Process the images
input_data = [process_image(path, input_shape) for path in sample_image_paths]
print(len(input_data))
print(input_data[0].shape)

# Upload dataset (in the same order)
qai_hub.upload_dataset({"image": input_data})
