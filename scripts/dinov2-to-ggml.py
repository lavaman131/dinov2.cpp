import argparse
import struct
from typing import Dict, BinaryIO
import numpy as np
import torch
from transformers import AutoModel, AutoConfig, Dinov2ForImageClassification

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
    fname_out = f"./ggml-model-{['f32', 'f16'][args.ftype]}.gguf"

    # Load the pretrained model
    # model = AutoModel.from_pretrained(args.model_name)
    model = Dinov2ForImageClassification.from_pretrained(args.model_name)
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

    # Write to file
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", GGML_MAGIC))  # Magic: ggml in hex
        for param in hparams.values():
            fout.write(struct.pack("i", param))
        fout.write(struct.pack("i", args.ftype))

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
            process_and_write_variable(fout, k, v, args.ftype)

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
    ftype_cur = (
        1
        if ftype == 1 and tensor.ndim != 1 and name not in {"pos_embed", "cls_token", "register_tokens"}
        else 0
    )
    data = data.astype(np.float32) if ftype_cur == 0 else data.astype(np.float16)

    if name == "patch_embed.proj.bias":
        data = data.reshape(1, data.shape[0], 1, 1)

    str_name = name.encode("utf-8")
    file.write(struct.pack("iii", len(data.shape), len(str_name), ftype_cur))
    for dim_size in reversed(data.shape):
        file.write(struct.pack("i", dim_size))
    file.write(str_name)
    data.tofile(file)


if __name__ == "__main__":
    main()
