#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml/examples/stb_image.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <algorithm>

struct dino_hparams {
    int32_t hidden_size = 768;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t num_classes = 1000;
    int32_t patch_size = 8;
    int32_t img_size = 224;
    int32_t ftype = 1;
    float eps = 1e-6f;
    std::string interpolation = "bicubic";
    std::map<int, std::string> id2label;

    int32_t n_enc_head_dim() const;

    int32_t n_img_size() const;

    int32_t n_patch_size() const;

    int32_t n_img_embd() const;
};

struct dino_block {
    struct ggml_tensor *norm1_w;
    struct ggml_tensor *norm1_b;
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;
    struct ggml_tensor *dense_w;
    struct ggml_tensor *dense_b;
    struct ggml_tensor *layer_scale1_lam;
    struct ggml_tensor *norm2_w;
    struct ggml_tensor *norm2_b;
    struct ggml_tensor *fc1_w;
    struct ggml_tensor *fc1_b;
    struct ggml_tensor *fc2_w;
    struct ggml_tensor *fc2_b;
    struct ggml_tensor *layer_scale2_lam;
};

struct classifier_head {
    struct ggml_tensor *norm_w;
    struct ggml_tensor *norm_b;
    struct ggml_tensor *head_w;
    struct ggml_tensor *head_b;
};

struct dino_image_encoder {
    struct ggml_tensor *pos_embed;
    struct ggml_tensor *cls_token;
    struct ggml_tensor *patch_embed_w;
    struct ggml_tensor *patch_embed_b;
    std::vector<dino_block> layers;
};

struct dino_state {
    struct ggml_tensor *prediction;
    struct ggml_tensor *patch_tokens;
    struct ggml_context *ctx;
    std::vector<uint8_t> work_buffer;
    std::vector<uint8_t> buf_alloc_img_enc;
    std::vector<uint8_t> buf_compute_img_enc;
    ggml_gallocr_t allocr = {};
};

struct dino_model {
    dino_hparams hparams;
    dino_image_encoder enc_img;
    classifier_head classifier;
    struct ggml_context *ctx;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct image_u8 {
    int nx;
    int ny;
    std::vector<uint8_t> data;
};

struct image_f32 {
    int nx;
    int ny;
    std::vector<float> data;
};

struct dino_params {
    int32_t seed = -1;
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t topk = 5;
    std::string model = "../ggml-model-f16.gguf"; // model path
    std::string fname_inp = "../assets/tench.jpg"; // image path
    float eps = 1e-6f; // epsilon used in LN
};

void print_t_f32(const char *title, const struct ggml_tensor *t, int n);

static void ggml_disconnect_node_from_graph(ggml_tensor *t);

void ggml_graph_compute_helper(std::vector<uint8_t> &buf, ggml_cgraph *graph, int n_threads);

bool load_image_from_file(const std::string &fname, image_u8 &img);

bool dino_image_preprocess(const image_u8 &img, image_f32 &res, const dino_hparams &params);

bool dino_model_load(const std::string &fname, dino_model &model);

struct ggml_cgraph *dino_encode_image(const dino_model &model, dino_state &state, const image_f32 &img);

int dino_predict(const dino_model &model, dino_state &state, const image_f32 img1, const dino_params &params,
                 std::vector<std::pair<float, int> > &predictions);

void print_usage(int argc, char **argv, const dino_params &params);

bool dino_params_parse(int argc, char **argv, dino_params &params);
