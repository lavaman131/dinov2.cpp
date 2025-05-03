# dinov2.cpp

DINOv2 pretrained visual models in C/C++ using ggml and OpenCV.

## Description

This project provides an implementation of the DINOv2 family of models in C++. These foundation models have been pretrained
for image-level and pixel-level visual tasks, and provide a broad range of possible applications in image analysis. We aim 
to provide all the functionalities available in the [pytorch implementation](https://github.com/facebookresearch/dinov2) in C++.
This lightweight version of DINOv2 is intended to reduce inference time and required memory, using [ggml](https://github.com/ggerganov/ggml)
and [OpenCV](https://github.com/opencv/opencv), particularly for use on edge devices. This implementation was heavily inspired by and built on 
existing code from [vit.cpp](https://github.com/staghado/vit.cpp).


<details>
<summary>Table of Contents</summary>

- [dinov2.cpp](#dinov2cpp)
    - [Description](#description)
    - [Features](#features)
    - [DINOv2 Overview](#dinov2-overview)
      - [Quick example](#quick-example)
        - [Feature Extraction](#feature-extraction)
        - [Classification Output](#classification-output)
      - [Realtime Demo](#realtime-demo)
    - [Convert PyTorch to GGUF](#convert-pytorch-to-gguf)
    - [Build](#build)
        - [Install OpenCV](#install-opencv)
          - [Configure Environment Variables](#configure-environment-variables)
        - [Simple build](#simple-build)
            - [inference.cpp (Classification)](#inferencecpp-classification)
            - [inference.cpp (Feature Extraction)](#inferencecpp-feature-extraction)
            - [realtime.cpp (Live Feature Extraction)](#realtimecpp-live-feature-extraction)
        - [Per device optimizations](#per-device-optimizations)
            - [For AMD host processors](#for-amd-host-processors)
        - [Using OpenMP](#using-openmp)
    - [Run](#run)
      - [inference.cpp](#inferencecpp)
      - [realtime.cpp](#realtimecpp)
    - [Benchmark against PyTorch](#benchmark-against-pytorch)
        - [DINOv2 inference](#dinov2-inference)
          - [DINOv2 with Register Tokens](#dinov2-with-register-tokens)
          - [DINOv2 without Register Tokens](#dinov2-without-register-tokens)
        - [Benchmark on your machine](#benchmark-on-your-machine)
    - [Quantization](#quantization)
        - [Results](#results)
    - [To-Do List](#to-do-list)

</details>

## Features

- Dependency-free and lightweight inference thanks to [ggml](https://github.com/ggerganov/ggml).
- Support for DINO models from huggingface with conversion from pytorch weights to gguf.
- 4-bit, 5-bit and 8-bit quantization support.


## DINOv2 Overview

The implemented architecture is based on the DINOv2 architecture:

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## Quick example

#### Feature Extraction

  <p align="center">
    <img src="assets/tench.jpg" alt="example input" width="50%" height="auto">
  </p>

  <p align="center">
    <img src="assets/pca_visual.jpg" alt="PCA output" width="50%" height="auto">
  </p>

#### Classification Output
  <pre>
  $ ./bin/dinov2 -t 4 -m ../ggml-model.gguf -i ../assets/tench.jpg 
  main: seed = 42
  main: loaded image '../assets/tench.jpg' (408 x 612)
  dino_model_load: loading model from '../ggml-model.gguf' - please wait
  dino_model_load: hidden_size            = 384
  dino_model_load: num_hidden_layers      = 12
  dino_model_load: num_register_tokens    = 4
  dino_model_load: num_attention_heads    = 6
  dino_model_load: patch_size             = 14
  dino_model_load: img_size               = 518
  dino_model_load: ftype                  = 1
  dino_model_load: qntvr                  = 0
  dino_model_load: num_classes            = 1000
  main: preprocessed image (224 x 224)


&gt; tench, Tinca tinca : 0.90
&gt; coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch : 0.05
&gt; goldfish, Carassius auratus : 0.01
&gt; suit, suit of clothes : 0.01
&gt; barracouta, snoek : 0.00

main: graph computation took 349 ms
  </pre>

## Realtime Demo

TODO add video

## Convert PyTorch to GGUF

```bash
# clone the repo recursively
git clone --recurse-submodules git@github.com:lavaman131/dinov2.cpp.git

cd dinov2.cpp

uv venv

# for MacOS/Linux
source .venv/bin/activate
# for Windows
.venv\Scripts\activate

uv sync --frozen

# convert the weights to gguf : dinov2 small with patch size of 14 and an image 
# size of 518  
# DINOv2 weights are always fp16
# without registers
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-small-imagenet1k-1-layer
# with registers
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-with-registers-small-imagenet1k-1-layer

```

## Build

### Install OpenCV

Refer to instructions on the [OpenCV](https://opencv.org/get-started/) website to install OpenCV on your machine.

<p align="center">
    <img src="assets/readme-assets/OpenCV-table.png">
  </p>

Using this table, pick your Operating System, and choose if you are going to build from source or install a prebuilt version. It is recommended to build from source, as prebuilt versions only support Visual Studio. OpenCV provides precise step by step instructions on how to build from source.

#### Configure Environment Variables

Once you have built OpenCV, you need to configure your environment to locate it. You have two options:

##### Option 1: Set Path in CMakeLists.txt
Add the following line to your CMakeLists.txt file:
```cmake
set(OpenCV_DIR /path/to/your/opencv/build/folder)
```
Replace `/path/to/your/opencv/build/folder` with the absolute path to your OpenCV build directory.

##### Option 2: Set System Environment Variables
Alternatively, configure your system environment variables:

1. Set `OpenCV_DIR` environment variable to the absolute path of your OpenCV build folder
2. Add the following directories to your system `PATH` variable:
   - The absolute path to the OpenCV `bin` folder
   - The absolute path to the OpenCV `lib` folder
   
Note: The `bin` and `lib` folders are typically located in the same directory.

### Simple Build
Add the `-c` flag when running inference.cpp to return the output predictions. Omitting the flag (by default) will return the patch
tokens.

#### inference.cpp (Classification)
```bash
# on MacOS/Linux 
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 4
./bin/inference -m ../ggml-model.gguf -i ../assets/tench.jpg -c
```

```bash
# on Windows
mkdir build ; cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
ninja
./bin/inference.exe -m ../ggml-model.gguf -i ../assets/tench.jpg -c
```
#### inference.cpp (Feature Extraction)
```bash
# on MacOS/Linux 
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 4
./bin/inference -m ../ggml-model.gguf -i ../assets/tench.jpg
```

```bash
# on Windows
mkdir build ; cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
ninja
./bin/inference.exe -m ../ggml-model.gguf -i ../assets/tench.jpg
```

#### realtime.cpp (Live Feature Extraction)
```bash
# on MacOS/Linux 
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 4
./bin/realtime -m ../ggml-model.gguf -i ../assets/tench.jpg
```

```bash
# on Windows
mkdir build ; cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
ninja
./bin/realtime.exe -m ../ggml-model.gguf -i ../assets/tench.jpg
```
The optimal number of threads to use depends on many factors and more is not always better. Usually using a number of
threads equal to the number of available physical cores gives the best performance in terms of speed.

### Per device optimizations

Generate per-device instructions that work best for the given machine rather than using general CPU instructions.

This can be done by specifying `-march=native` in the compiler flags.

* Multi-threading and vectorization
* Loop transformations(unrolling)

#### For AMD host processors

You can use a specialized compiler released by AMD to make full use of your specific processor's architecture.

Read more here : [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html)

You can follow the given instructions to install the AOCC compiler.

Please note that modern processors tend to see the greatest benefits from a specialized compiler, whereas older CPUs may experience little to no performance improvement.

### Using OpenMP

Additionally compile with OpenMP by specifying the `-fopenmp` flag to the compiler in the CMakeLists file,
allowing multithreaded runs. Make sure to also enable multiple threads when running, e.g.:

```bash
OMP_NUM_THREADS=4 ./bin/inference -t 4 -m ../ggml-model.bin -i ../assets/tench.jpg
```

## Run

#### inference.cpp
```bash
usage: ./bin/inference [options]

options:
  -h, --help              show this help message and exit
  -m FNAME, --model       model path (default: ../ggml-model.gguf)
  -i FNAME, --inp         input file (default: ../assets/tench.jpg)
  -o FNAME, --out         output file for backbone PCA features (default: pca_visual.png)
  -k N, --topk            top k classes to print (default: 5)
  -t N, --threads         number of threads to use during computation (default: 4)
  -c, --classify          whether to classify the image or get backbone PCA features (default: 0)
  -fa, --flash_attn       whether to enable flash_attn, less accurate (default: 0)
```

#### realtime.cpp
```bash
usage: ./bin/realtime [options]

options:
  -h, --help              show this help message and exit
  -m FNAME, --model       model path (default: ../ggml-model.gguf)
  -t N, --threads         number of threads to use during computation (default: 4)
  -fa, --flash_attn       whether to enable flash_attn, less accurate (default: 0)
  -cid, --camera_id       the idea of the camera for realtime backbone PCA feature streaming (default: 0)
```

## Benchmark against PyTorch

First experiments on Intel Core i9-14900HX show inference speedups (up to 3x faster for small model, ~1.5-2x faster for the rest) compared to native PyTorch inference.

### DINOv2 inference

You can efficiently run DINOv2 inference on the CPU.

Memory requirements and inference speed on Intel Core i9-14900HX (24 cores, 32 threads) for both native PyTorch and `dinov2.cpp`.
Using a thread count greater than 10 provides marginal improvements, but 24 threads were used for these runs. The reported results of inference speed correspond to 100 runs
averages for both PyTorch and `dinov2.cpp`.

#### DINOv2 with Register Tokens
| Model | Max Mem(PyTorch) |   Max Mem   | Speed(PyTorch) |    Speed    |
|:-----:|:----------------:|:-----------:|:--------------:|:-----------:|
| small  |     ~457 MB      | **~109 MB**  |     297 ms     | **96 ms**  |
| base |     ~720 MB      | **~367 MB**  |     436 ms     | **290 ms**  |
| large  |     ~1.57 GB     | **~1.2 GB** |    1331 ms     | **868 ms** |
| giant |     ~4.8 GB     | **~4.4 GB** |    4472 ms     | **2935 ms** |

> **Note:** The models used are of the form `dinov2-with-registers-{size}-imagenet1k-1-layer`

#### DINOv2 without Register Tokens
| Model | Max Mem(PyTorch) |   Max Mem   | Speed(PyTorch) |    Speed    |
|:-----:|:----------------:|:-----------:|:--------------:|:-----------:|
| small  |     ~455 MB      | **~110 MB**  |     181 ms     | **104 ms**  |
| base |     ~720 MB      | **~365 MB**  |     462 ms     | **292 ms**  |
| large  |     ~1.55 GB     | **~1.2 GB** |    1288 ms     | **864 ms** |
| giant |     ~4.8 GB     | **~4.4 GB** |    4384 ms     | **2938 ms** |

> **Note:** The models used are of the form `dinov2-{size}-imagenet1k-1-layer`.

### Benchmark on your machine

In order to test the inference speed on your machine, you can run the following scripts:

```bash
chmod +x scripts/benchmark.*

# install memory_profiler & threadpoolctl
pip install memory_profiler threadpoolctl

# run the benchmark of PyTorch
python scripts/benchmark.py

# run the benchmark of dinov2.cpp for non-quantized model
./scripts/benchmark.sh

# to run the benchamrk for quantized models; 4 threads and quantize flag
./scripts/benchmark.sh 4 1
```

Both scripts use 4 threads by default. In Python, the `threadpoolctl` library is used to limit the number of threads
used by PyTorch.

## Quantization

`dinov2.cpp` supports quantization strategies from ggml such as q4_0, q4_1, q5_0, q5_1 and q8_0 types.
You can quantize a model in F32 (the patch embedding is in F16) to one of these types by using the `./bin/quantize`
binary.

```
usage: ./bin/quantize /path/to/ggml-model.gguf /path/to/ggml-model-quantized.gguf type                              
  type = 2 - q4_0                                                                                                       
  type = 3 - q4_1                                                                                                       
  type = 6 - q5_0                                                                                                       
  type = 7 - q5_1                                                                                                       
  type = 8 - q8_0                                                                                                       
```

For example, you can run the following to convert the model to q5_1:

```shell
./bin/quantize ../ggml-model.gguf ../ggml-model-quant.gguf 7
```

Then you can use `ggml-model-quant.gguf` just like the model in F16.

### Results

Here are the benchmarks for the different models and quantizations on my machine:
For accurate estimation of run times, these benchmarks were run 100 times each.

#### DINOv2 with Register Tokens
| Model  | Quantization | Speed (ms) | Mem (MB) |
| :----: | :----------: | :--------: | :------: |
| small | q4_0 | 90 | 49 |
| small | q4_1 | 78 | 51 |
| small | q5_0 | 86 | 54 |
| small | q5_1 | 88 | 57 |
| small | q8_0 | 82 | 70 |
| base | q4_0 | 176 | 129 |
| base | q4_1 | 190 | 139 |
| base | q5_0 | 218 | 150 |
| base | q5_1 | 203 | 160 |
| base | q8_0 | 188 | 212 |
| large | q4_0 | 508 | 371 |
| large | q4_1 | 516 | 407 |
| large | q5_0 | 618 | 443 |
| large | q5_1 | 615 | 480 |
| large | q8_0 | 507 | 661 |
| giant | q4_0 | 1600 | 1281 |
| giant | q4_1 | 1617 | 1417 |
| giant | q5_0 | 2060 | 1552 |
| giant | q5_1 | 2036 | 1688 |
| giant | q8_0 | 1598 | 2364 |

#### DINOv2 without Register Tokens

| Model  | Quantization | Speed (ms) | Mem (MB) |
| :----: | :----------: | :--------: | :------: |
| small | q4_0 | 91 | 49 |
| small | q4_1 | 82 | 51 |
| small | q5_0 | 83 | 54 |
| small | q5_1 | 91 | 57 |
| small | q8_0 | 93 | 70 |
| base | q4_0 | 179 | 129 |
| base | q4_1 | 187 | 140 |
| base | q5_0 | 218 | 150 |
| base | q5_1 | 207 | 160 |
| base | q8_0 | 193 | 212 |
| large | q4_0 | 509 | 371 |
| large | q4_1 | 503 | 407 |
| large | q5_0 | 617 | 444 |
| large | q5_1 | 615 | 480 |
| large | q8_0 | 527 | 661 |
| giant | q4_0 | 1586 | 1282 |
| giant | q4_1 | 1627 | 1417 |
| giant | q5_0 | 2036 | 1552 |
| giant | q5_1 | 2075 | 1688 |
| giant | q8_0 | 1554 | 2365 |

This project was built on and highly inspired by vit.cpp:

* [vit.cpp](https://github.com/staghado/vit.cpp)

