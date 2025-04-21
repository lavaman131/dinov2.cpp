import time
from typing import Tuple, Final, Sequence, Dict, Any, Union
import torch
from torchvision.transforms import v2
from memory_profiler import memory_usage
from PIL import Image
from threadpoolctl import threadpool_limits
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import torch.nn as nn
import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

IMAGENET_DEFAULT_MEAN: Final[Sequence[float]] = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD: Final[Sequence[float]] = (0.229, 0.224, 0.225)


def make_normalize_transform(
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
        *,
        resize_size: int = 256,
        interpolation=v2.InterpolationMode.BICUBIC,
        crop_size: int = 224,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [
        v2.Resize(resize_size, interpolation=interpolation),
        v2.CenterCrop(crop_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        make_normalize_transform(mean=mean, std=std),
    ]
    return v2.Compose(transforms_list)


@torch.inference_mode
def predict(inputs: Dict[str, Any], model: nn.Module) -> torch.Tensor:
    logits = model(**inputs).logits
    return logits


def benchmark_model(
        image_path: str, model_name: str, n: int, device: str
) -> Tuple[float, float]:
    times = []
    peak_memory_usages = []

    model = AutoModelForImageClassification.from_pretrained(model_name, device_map=device)
    preprocess = make_classification_eval_transform()
    with Image.open(image_path).convert("RGB") as image:
        inputs = dict(pixel_values=preprocess(image).to(device, non_blocking=True).unsqueeze_(0))

    for _ in range(n):
        # Measure peak memory usage
        # peak_memory_usage = memory_usage(
        #     (predict, (inputs, model)),
        #     interval=0.01,
        #     max_usage=True,
        #     include_children=True,
        # )
        # peak_memory_usages.append(peak_memory_usage)

        start_time = time.perf_counter_ns()
        predict(inputs, model)
        end_time = time.perf_counter_ns()

        time_taken = end_time - start_time
        times.append(time_taken)

    avg_time = sum(times) / n * 1e-6  # Convert to milliseconds
    max_peak_memory = sum(peak_memory_usages) / n
    return avg_time, max_peak_memory


def main() -> None:
    # model variants
    model_variants = {
        "small": "facebook/dinov2-small-imagenet1k-1-layer",
        "base": "facebook/dinov2-base-imagenet1k-1-layer",
        "large": "facebook/dinov2-large-imagenet1k-1-layer",
        "giant": "facebook/dinov2-giant-imagenet1k-1-layer",
    }

    # an image
    image_path = "./assets/tench.jpg"
    device = "mps"
    n = 100

    with threadpool_limits(limits=4):
        print("| Model | Speed (ms)   |   Mem (MB)       |")
        print("|-------|--------------|------------------|")

        for name, model_name in model_variants.items():
            avg_time, peak_memory = benchmark_model(image_path, model_name, n, device)
            print(f"| {name} | {avg_time:.0f} | {peak_memory:.0f} |")


if __name__ == "__main__":
    main()
