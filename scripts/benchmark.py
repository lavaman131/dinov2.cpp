import os
import time
from typing import Tuple, Final, Sequence, Dict, Any, Union
import torch
from torchvision.transforms import v2
from PIL import Image
from threadpoolctl import threadpool_limits
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import torch.nn.functional as F
import torch.nn as nn
import logging
import resource  # Import resource for CPU memory measurement
import sys  # Added for platform check

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


@torch.inference_mode()
def predict(inputs: Dict[str, Any], model: nn.Module) -> torch.Tensor:
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    return probs

    # Use appropriate synchronization


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # No sync needed for CPU


def benchmark_model(
        image_path: str, model_name: str, n: int, device: torch.device
) -> Tuple[float, float]:
    times = []
    peak_memory_mb = 0.0  # Initialize peak memory

    # Reset peak memory stats before the main loop if using CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # No longer need start_rss for delta

    for _ in range(n):
        sync(device)
        start_time = time.perf_counter_ns()
        config = AutoConfig.from_pretrained(model_name, attn_implementation="sdpa")
        model = AutoModelForImageClassification.from_pretrained(
            model_name, device_map=device, config=config
        )
        model.eval()
        preprocess = make_classification_eval_transform()
        with Image.open(image_path).convert("RGB") as image:
            processed_image = preprocess(image).to(device, non_blocking=True).unsqueeze_(0)
            inputs = dict(pixel_values=processed_image)
        predict(inputs, model)
        sync(device)
        end_time = time.perf_counter_ns()

        time_taken = end_time - start_time
        times.append(time_taken)

    # Calculate peak memory usage *after* the loop
    if device.type == "cuda":
        # Peak GPU memory allocated by PyTorch tensors in MB
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    elif device.type == "mps":
        peak_memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
    elif device.type == "cpu":
        # Peak resident set size (RSS) of the *entire process* in MB
        # This is the high-water mark for the process, including model loading etc.
        # ru_maxrss units are Bytes on macOS, KiB on Linux.
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":  # macOS uses Bytes
            peak_memory_mb = peak_rss / (1024 * 1024)
        else:  # Assume Linux uses KiB
            peak_memory_mb = peak_rss / 1024

    # Calculate average time excluding the first run if it was warm-up
    valid_times = times[1:] if n > 1 and len(times) > 1 else times
    if not valid_times:  # Handle case where n=0 or n=1
        avg_time_ms = 0.0
    else:
        avg_time_ns = sum(valid_times) / len(valid_times)
        avg_time_ms = avg_time_ns * 1e-6  # Convert ns to ms

    return avg_time_ms, peak_memory_mb


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
    device = torch.device("cpu")  # Keep CPU for now as in original code
    n = 100  # Number of timed iterations (excluding warm-up)

    print(f"Benchmarking on device: {device}")
    print(f"Number of timed iterations: {n}")
    print("| Model Variant | Avg Speed (ms) | Peak RSS (MB)    |")  # Updated header
    print("|---------------|----------------|------------------|")
    for name, model_name in model_variants.items():
        avg_time, peak_memory = benchmark_model(image_path, model_name, n, device)
        print(f"| {name:<13} | {avg_time:>14.2f} | {peak_memory:>16.2f} |")


if __name__ == "__main__":
    # control number of threads for fair comparison, default is 1 => total threads = num_cores * num_threads_per_core
    num_threads_per_core = 1
    os.environ["OMP_NUM_THREADS"] = str(num_threads_per_core)
    torch.set_num_threads(num_threads_per_core)
    print(f"Total threads: {(os.cpu_count() or 1) * torch.get_num_threads()}")
    main()
