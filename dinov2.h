#pragma once


#include "ggml.h"
#include "ggml-alloc.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cinttypes>
#include <optional>
#include <memory>
#include <thread>
#include <opencv2/core/mat.hpp>

constexpr float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_DEFAULT_STD[3] = {0.229f, 0.224f, 0.225f};

uint32_t get_val_u32(const struct gguf_context *ctx,
                     const char *key);

const char *get_val_str(const struct gguf_context *ctx, const char *key);

struct dino_hparams {
    uint32_t hidden_size = 768;
    uint32_t num_hidden_layers = 12;
    uint32_t num_attention_heads = 12;
    uint32_t num_classes = 1000;
    uint32_t num_register_tokens = 0;
    uint32_t patch_size = 8;
    uint32_t img_size = 224;
    uint32_t ftype = 1;
    float eps = 1e-6f;
    std::string interpolation = "bicubic";
    std::map<int, std::string> id2label;

    uint32_t n_enc_head_dim() const;

    uint32_t n_img_size() const;

    uint32_t n_patch_size() const;

    uint32_t n_img_embd() const;
};


struct dino_model {
    dino_hparams hparams;
    struct ggml_context *ctx;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct dino_params {
    uint32_t seed = 42;
    uint32_t topk = 5;
    bool enable_flash_attn = false;
    uint8_t camera_id = 0; // camera id for realtime PCA feature streaming
    uint32_t n_threads = std::min(4u, std::thread::hardware_concurrency());;
    bool classify = false;
    std::string model = "../ggml-model-f16.gguf"; // model path
    std::string fname_inp = "../assets/tench.jpg"; // image path
    std::string image_out = "pca_visual.jpg"; // output of pca visualization (if used)
    float eps = 1e-6f; // epsilon used in LN
};

struct ggml_tensor *attn(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph,
                         const dino_model &model, const dino_params &params);

struct ggml_tensor *mlp(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph,
                        const dino_model &model, const dino_params &params);

struct ggml_tensor *swiglu_ffn(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph,
                               const dino_model &model, const dino_params &params);

void forward_features(cv::Size img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                      const dino_model &model, const dino_params &params);

void forward_head(cv::Size img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                  const dino_model &model, const dino_params &params);

struct dino_output {
    std::optional<std::vector<uint32_t> > preds;
    std::optional<cv::Mat> patch_tokens;
};

void print_t_f32(const char *title, const struct ggml_tensor *t, int n);

static void ggml_disconnect_node_from_graph(ggml_tensor *t);

cv::Mat dino_classify_preprocess(cv::Mat &img, cv::Size img_size, const dino_hparams &params);

cv::Mat dino_preprocess(cv::Mat &img, cv::Size img_size, const dino_hparams &params);

bool dino_model_load(cv::Size img_size, const std::string &fname, dino_model &model,
                     const dino_params &params);

std::vector<float> interpolate_pos_embed(cv::Size img_size,
                                         const float *pos_embed_data,
                                         const dino_hparams &hparams);

struct ggml_cgraph *build_graph(
    cv::Size img_size,
    struct ggml_context *ctx_cgraph,
    const dino_model &model,
    const dino_params &params);

std::unique_ptr<dino_output> dino_predict(const dino_model &model, const cv::Mat &img,
                                          const dino_params &params);

void print_usage(int argc, char **argv, const dino_params &params);

bool dino_params_parse(int argc, char **argv, dino_params &params);

bool dino_model_quantize(const std::string& fname_inp, const std::string& fname_out, int itype);