import argparse
import struct
from typing import Dict, BinaryIO
import numpy as np
import torch
from transformers import AutoModel, AutoConfig, AutoModelForImageClassification

GGML_MAGIC = 0x67676d6c


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


def main() -> None:
    args = get_args()
    # Output file name
    fname_out = f"./ggml-model-f16.gguf"

    # Load the pretrained model
    # model = AutoModel.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    id2label = config.id2label

    # Hyperparameters
    hparams = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_classes": len(id2label),
        "patch_size": config.patch_size,
        "img_size": config.image_size,
    }

    ftype = 1  # float16

    # Write to file
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", GGML_MAGIC))  # Magic: ggml in hex
        for param in hparams.values():
            fout.write(struct.pack("i", param))
        fout.write(struct.pack("i", ftype))

        # Write id2label dictionary to the file
        write_id2label(fout, id2label)

        # Process and write model weights
        for k, v in model.state_dict().items():
            if k.startswith("norm_pre"):
                print(f"the model {args.model_name} contains a pre_norm")
                print(k)
                continue
            print(
                "Processing variable: " + k + " with shape: ",
                v.shape,
                " and type: ",
                v.dtype,
            )
            process_and_write_variable(fout, k, v, ftype)

        print("Done. Output file: " + fname_out)


def write_id2label(file: BinaryIO, id2label: Dict[str, str]) -> None:
    file.write(struct.pack("i", len(id2label)))
    for key, value in id2label.items():
        file.write(struct.pack("i", key))
        encoded_value = value.encode("utf-8")
        file.write(struct.pack("i", len(encoded_value)))
        file.write(encoded_value)


def process_and_write_variable(file: BinaryIO, name: str, tensor: torch.Tensor, ftype: int) -> None:
    data = tensor.numpy()

    name_without_prefix = name

    if name_without_prefix.startswith("dinov2"):
        name_without_prefix = ".".join(name.split(".")[1:])

    if name_without_prefix == "embeddings.mask_token":
        # Skip the mask token
        return

    data = data.astype(np.float32) if ftype == 0 else data.astype(np.float16)

    if name_without_prefix == "embeddings.patch_embeddings.projection.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    str_name = name_without_prefix.encode("utf-8")
    file.write(struct.pack("iii", len(data.shape), len(str_name), ftype))
    for dim_size in reversed(data.shape):
        file.write(struct.pack("i", dim_size))
    file.write(str_name)
    data.tofile(file)


if __name__ == "__main__":
    main()
