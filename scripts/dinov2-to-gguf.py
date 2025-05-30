import argparse
import struct
from typing import Dict, BinaryIO, Final
import numpy as np
import torch
from transformers import AutoModel, AutoConfig, AutoModelForImageClassification
from gguf import GGUFWriter, GGUFEndian, GGMLQuantizationType
from dinov2_inference.types import GGMLNumpyType


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
    args = parser.parse_args()
    return args


ARCH: Final[str] = "dinov2"


@torch.no_grad()
def main() -> None:
    args = get_args()
    # Output file name
    fname_out = f"./ggml-model.gguf"

    is_classifier = "imagenet" in args.model_name

    # Load the pretrained model
    id2label = {}
    config = AutoConfig.from_pretrained(args.model_name)
    if is_classifier:
        model = AutoModelForImageClassification.from_pretrained(args.model_name)
        id2label = config.id2label
    else:
        model = AutoModel.from_pretrained(args.model_name)

    ggml_type = GGMLQuantizationType.F16

    # Hyperparameters
    hparams = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_classes": len(id2label),
        "patch_size": config.patch_size,
        "img_size": config.image_size,
        "ftype": ggml_type.value
    }

    gguf_writer = GGUFWriter(
        path=fname_out,
        arch=ARCH,
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
        save_tensor(gguf_writer, k, v, ggml_type)

    if num_register_tokens > 0:
        layers = model.dinov2_with_registers.encoder.layer
    else:
        layers = model.dinov2.encoder.layer

    for i, layer in enumerate(layers):
        base_name = f"encoder.layer.{i}.attention.attention"

        q = layer.attention.attention.query.weight
        k = layer.attention.attention.key.weight
        v = layer.attention.attention.value.weight
        qkv = torch.cat([q, k, v], dim=0)
        name = base_name + ".qkv.weight"
        print(
            "Processing variable: " + name + " with shape: ",
            qkv.shape,
            " and type: ",
            qkv.dtype,
        )
        save_tensor(gguf_writer, name, qkv, ggml_type)

        q = layer.attention.attention.query.bias
        k = layer.attention.attention.key.bias
        v = layer.attention.attention.value.bias
        qkv = torch.cat([q, k, v], dim=0)
        name = base_name + ".qkv.bias"
        print(
            "Processing variable: " + name + " with shape: ",
            qkv.shape,
            " and type: ",
            qkv.dtype,
        )
        save_tensor(gguf_writer, name, qkv, ggml_type)

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
        writer: GGUFWriter, name: str, tensor: torch.Tensor, ggml_type: GGMLQuantizationType
) -> None:
    data = tensor.numpy()

    if tensor.ndim == 1 or name in {
        "embeddings.position_embeddings",
        "embeddings.cls_token",
        "embeddings.register_tokens",
    }:
        ggml_type = GGMLQuantizationType.F32

    np_dtype = GGMLNumpyType[ggml_type.name].value

    data = data.astype(np_dtype)

    if name == "embeddings.patch_embeddings.projection.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    writer.add_tensor(name, data)

    print(name, data.shape, ggml_type.name)


def get_tensor_name(name: str) -> str:
    if name.startswith(ARCH):
        name = ".".join(name.split(".")[1:])
    return name


def should_skip_tensor(name: str) -> bool:
    return name in {"embeddings.mask_token"} or name.startswith(
        "norm_pre"
    ) or "attention.attention" in name


if __name__ == "__main__":
    main()
