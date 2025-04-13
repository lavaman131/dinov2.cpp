import time
from typing import Tuple, Final, Sequence
import torch
from torchvision.transforms import v2
from memory_profiler import memory_usage
from PIL import Image
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

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


def process_and_predict(image_path: str, model_path: str) -> torch.Tensor:
    model = AutoModel.from_pretrained(model_path)
    preprocess = make_classification_eval_transform()

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze_(0)

    with torch.inference_mode():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=-1)

    return probabilities


def benchmark_model(image_path: str, model_name: str, N: int = 10) -> Tuple[float, float]:
    times = []
    peak_memory_usages = []

    for _ in range(N):
        start_time = time.time()

        # Measure peak memory usage
        peak_memory_usage = memory_usage(
            (process_and_predict, (image_path, model_name)),
            interval=0.01,
            max_usage=True,
            include_children=True,
        )

        end_time = time.time()

        time_taken = end_time - start_time
        times.append(time_taken)
        peak_memory_usages.append(peak_memory_usage)

    avg_time = sum(times) / N * 1000  # in ms
    max_peak_memory = sum(peak_memory_usages) / N
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
    with threadpool_limits(limits=4):
        print("| Model | Speed (ms)   |   Mem (MB)       |")
        print("|-------|--------------|------------------|")

        for name, model_name in model_variants.items():
            avg_time, peak_memory = benchmark_model(image_path, model_name)
            print(f"| {name} | {avg_time:.0f} | {peak_memory:.0f} |")


if __name__ == "__main__":
    main()
