import argparse
import struct
from typing import Dict, BinaryIO
import numpy as np
import torch
from transformers import AutoModel, AutoConfig, AutoModelForImageClassification
from gguf import GGUFWriter, GGUFEndian

DATA_TYPES = ["f32", "f16"]


def get_args() -> argparse.Namespace:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights of a Vision Transformer to the ggml file format."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-small-imagenet1k-1-layer",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--ftype",
        type=int,
        choices=[0, 1],
        default=1,
        help="float type: 0 for float32, 1 for float16",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    # Output file name
    fname_out = f"./ggml-model-{DATA_TYPES[args.ftype]}.gguf"

    # Load the pretrained model
    model = AutoModel.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    id2label = {}
    if "imagenet" in args.model_name:
        id2label = config.id2label

    # Hyperparameters
    hparams = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_classes": len(id2label),
        "patch_size": config.patch_size,
        "img_size": config.image_size,
        "ftype": args.ftype
    }

    gguf_writer = GGUFWriter(
        path=fname_out,
        arch="dinov2",
    )

    # Write id2label dictionary to the file
    write_id2label(gguf_writer, id2label)

    num_register_tokens = 0
    # Process and write model weights
    for k, v in model.state_dict().items():
        k = get_tensor_name(k)
        if should_skip_tensor(k):
            continue
        elif k == "embeddings.register_tokens":
            num_register_tokens = v.shape[1]
        print(
            "Processing variable: " + k + " with shape: ",
            v.shape,
            " and type: ",
            v.dtype,
        )
        save_tensor(gguf_writer, k, v, args.ftype)

    hparams["num_register_tokens"] = num_register_tokens

    print(hparams)
    write_hparams(gguf_writer, hparams)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print("Done. Output file: " + fname_out)


def write_id2label(writer: GGUFWriter, id2label: Dict[int, str]) -> None:
    for key, value in id2label.items():
        writer.add_string(str(key), value)


def write_hparams(writer: GGUFWriter, hparams: Dict[str, int]) -> None:
    for key, value in hparams.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        else:
            raise ValueError(f"Unsupported hyperparameter type: {type(value)}")


def save_tensor(
        writer: GGUFWriter, name: str, tensor: torch.Tensor, ftype: int
) -> None:
    data = tensor.numpy()

    ftype = (
        1
        if ftype == 1
           and tensor.ndim != 1
           and name
           not in {
               "embeddings.position_embeddings",
               "embeddings.cls_token",
               "embeddings.register_tokens",
           }
        else 0
    )
    data = data.astype(np.float32) if ftype == 0 else data.astype(np.float16)

    if name == "embeddings.patch_embeddings.projection.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    writer.add_tensor(name, data)

    print(name, data.shape, DATA_TYPES[ftype])


def get_tensor_name(name: str) -> str:
    if name.startswith("dinov2"):
        name = ".".join(name.split(".")[1:])
    return name


def should_skip_tensor(name: str) -> bool:
    return name in {"embeddings.mask_token"} or name.startswith(
        "norm_pre"
    )


if __name__ == "__main__":
    main()
