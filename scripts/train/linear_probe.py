import argparse
import datetime
import json
import math
import numpy as np
import os
import time
from pathlib import Path
import webdataset as wds
import torch
import torch.backends.cudnn as cudnn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from efficient_cv.utils.train import get_config

# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from timm.layers.weight_init import trunc_normal_

from efficient_cv.data.imagenet import (
    NUM_IMAGENET_TRAIN_SAMPLES,
    NUM_IMAGENET_VAL_SAMPLES,
)
import efficient_cv.utils.misc as misc
from efficient_cv.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from efficient_cv.utils.lars import LARS
from efficient_cv.data.crop import RandomResizedCrop

from efficient_cv.train.finetune import train_one_epoch, evaluate

import types

from efficient_cv.models.titok import TiTok

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def main():
    config = get_config()
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(config)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(config).replace(", ", ",\n"))

    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + misc.get_rank()
    misc.set_seed(seed)

    # linear probe: weak augmentation
    transform_train = transforms.Compose(
        [
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    log_writer = None
    if global_rank == 0 and config.log_dir is not None and not config.eval:
        os.makedirs(config.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=config.log_dir) # Don't have tensorboard

    # WebDataset transformations
    def transform_train_sample(sample):
        return transform_train(sample["jpg"]), sample["cls"]

    def transform_val_sample(sample):
        return transform_val(sample["jpg"]), sample["cls"]

    # Set up WebDataset loaders
    train_pattern = os.path.join(config.data_path, config.train_pattern)
    val_pattern = os.path.join(config.data_path, config.val_pattern)

    # Create WebDataset for training
    dataset_train = wds.WebDataset(
        train_pattern,
        resampled=True,
        shardshuffle=False,
        nodesplitter=wds.split_by_node if num_tasks > 1 else None,
    )
    dataset_train = dataset_train.decode("pil").map(transform_train_sample)
    dataset_train = dataset_train.batched(config.batch_size)

    # Create WebDataset for validation
    dataset_val = wds.WebDataset(
        val_pattern,
        shardshuffle=False,
        nodesplitter=wds.split_by_node
        if (num_tasks > 1 and config.dist_eval)
        else None,
    )
    dataset_val = dataset_val.decode("pil").map(transform_val_sample)
    dataset_val = dataset_val.batched(config.batch_size)

    print(f"Creating WebDataset loaders: {train_pattern} and {val_pattern}")

    # Calculate epoch size for WebDataset
    GLOBAL_BATCH_SIZE = config.batch_size * num_tasks
    ONE_EPOCH_TRAIN = math.ceil(
        NUM_IMAGENET_TRAIN_SAMPLES / (GLOBAL_BATCH_SIZE * config.num_workers)
    )
    ONE_EPOCH_VAL = math.ceil(
        NUM_IMAGENET_VAL_SAMPLES / (GLOBAL_BATCH_SIZE * config.num_workers)
    )

    # Create WebLoaders
    data_loader_train = (
        wds.WebLoader(
            dataset_train,
            batch_size=None,  # already batched
            num_workers=config.num_workers,
            pin_memory=config.pin_mem,
            drop_last=False,
        )
        .with_epoch(ONE_EPOCH_TRAIN)
        .with_length(ONE_EPOCH_TRAIN)
    )

    data_loader_val = (
        wds.WebLoader(
            dataset_val,
            batch_size=None,  # already batched
            num_workers=config.num_workers,
            pin_memory=config.pin_mem,
            drop_last=False,
        )
        .with_epoch(ONE_EPOCH_VAL)
        .with_length(ONE_EPOCH_VAL)
    )

    model = TiTok.from_pretrained(config.pretrained_model_name_or_path)
    output_tokens = model.num_latent_tokens

    # For linear probing, modify model's head to have linear layer, batch norm also used by MAE
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(output_tokens, affine=False, eps=1e-6),
        torch.nn.Linear(output_tokens, config.nb_classes),
    )

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head[1].weight, std=0.01)

    # Change model's forwarding to only use encoder and the new linear layer
    def forward_linear_probe(x):
        z_quantized, result_dict = model.encode(x)  # B, 12, 1, 128

        # Global pooling, not sure if this is working as intended, might need more testing
        pooled = z_quantized.mean(dim=-1).squeeze_(-1)

        return model.head(pooled)

    model.forward = forward_linear_probe

    # Freeze all layers besides the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = config.batch_size * config.accum_iter * misc.get_world_size()

    if config.lr is None:  # only base_lr is specified
        config.lr = config.blr * eff_batch_size / 256

    print("base lr: %.2e" % (config.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % config.lr)

    print("accumulate grad iterations: %d" % config.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    model_without_ddp = model.module

    optimizer = LARS(
        model_without_ddp.head.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=config,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if config.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    print(f"Start training for {config.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(config.start_epoch, config.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=config,
        )
        if config.output_dir:
            misc.save_model(
                args=config,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if config.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(config.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
