#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cinttypes>

#include "../../../../../Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/System/Library/Frameworks/Security.framework/Versions/A/Headers/cssmconfig.h"

u_int32_t get_val_u32(const struct gguf_context *ctx,
                      const char *key);

const char *get_val_str(const struct gguf_context *ctx, const char *key);

struct dino_hparams {
    int32_t hidden_size = 768;
    int32_t num_hidden_layers = 12;
    int32_t num_attention_heads = 12;
    int32_t num_classes = 1000;
    int32_t num_register_tokens = 0;
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


struct dino_model {
    dino_hparams hparams;
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
    int32_t topk = 5;
    bool classify = false;
    std::string model = "../ggml-model-f16.gguf"; // model path
    std::string fname_inp = "../assets/tench.jpg"; // image path
    float eps = 1e-6f; // epsilon used in LN
};

void *forward_features(struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                       const dino_model &model, const dino_params &params);

void *forward_head(struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                   const dino_model &model, const dino_params &params);

struct dino_output {
    std::optional<std::vector<uint32> > preds;
    std::optional<std::vector<float> > patch_tokens;
};

void print_t_f32(const char *title, const struct ggml_tensor *t, int n);

static void ggml_disconnect_node_from_graph(ggml_tensor *t);

bool load_image_from_file(const std::string &fname, image_u8 &img);

bool dino_image_preprocess(const image_u8 &img, image_f32 &res, const dino_hparams &params);

bool dino_model_load(const std::string &fname, dino_model &model, const dino_params &params);

struct ggml_cgraph *build_graph(
    struct ggml_context *ctx_cgraph,
    const dino_model &model,
    const dino_params &params);

std::unique_ptr<dino_output> dino_predict(const dino_model &model, const image_f32 &img1,
                                          const dino_params &params);

void print_usage(int argc, char **argv, const dino_params &params);

bool dino_params_parse(int argc, char **argv, dino_params &params);
