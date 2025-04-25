#define CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>


#define STB_IMAGE_IMPLEMENTATION
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cinttypes>
#include <algorithm>
#include <iostream>

#include "ggml/src/ggml-impl.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

uint32_t dino_hparams::n_enc_head_dim() const {
    return hidden_size / num_attention_heads;
}

uint32_t dino_hparams::n_img_size() const {
    return img_size;
}

uint32_t dino_hparams::n_patch_size() const {
    return patch_size;
}

uint32_t dino_hparams::n_img_embd() const {
    return n_img_size() / n_patch_size();
}

uint32_t get_val_u32(const struct gguf_context *ctx,
                     const char *key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    assert(key_id >= 0);
    return gguf_get_val_u32(
        ctx, key_id);
}

const char *get_val_str(const struct gguf_context *ctx, const char *key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    assert(key_id >= 0);
    return gguf_get_val_str(ctx, key_id);
}

//
// Helpers
//

void print_t_f32(const char *title, const struct ggml_tensor *t, const int n = 10) {
    printf("%s\n", title);
    const auto *data = (float *) (t->data);
    printf("dims: % " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("First & Last %d elements:\n", n);
    for (int i = 0; i < std::min((int) (t->ne[0] * t->ne[1]), n); i++) {
        printf("%.5f ", data[i]);
        if (i != 0 && i % t->ne[0] == 0) {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < std::min((int) (t->ne[0] * t->ne[1]), n); i++) {
        printf("%.5f ", data[ggml_nelements(t) - n + i]);
        if ((ggml_nelements(t) - n + i) % t->ne[0] == 0) {
            printf("\n");
        }
    }
    printf("\n");
    double sum = 0.0;
    for (int i = 0; i < ggml_nelements(t); i++) {
        sum += data[i];
    }
    printf("sum:  %f\n\n", sum);
}

static void ggml_disconnect_node_from_graph(ggml_tensor *t) {
    t->op = GGML_OP_NONE;
    for (auto &i: t->src) {
        i = nullptr;
    }
}

cv::Mat dino_classify_preprocess(cv::Mat &img, const cv::Size img_size, const dino_hparams &params) {
    // 1) Convert to float and resize
    cv::Mat image;
    img.convertTo(image, CV_32FC3, 1.0 / 255.0);

    const auto new_size = cv::Size(256, 256);
    cv::resize(image, image,
               new_size,
               0, 0, cv::INTER_CUBIC);

    constexpr int crop_size = 224;
    const int offset_w = (image.cols - crop_size) / 2;
    const int offset_h = (image.rows - crop_size) / 2;
    const cv::Rect roi(offset_w, offset_h, crop_size, crop_size);
    image = image(roi);

    // 3) Channel-wise standardization
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - IMAGENET_DEFAULT_MEAN[2 - i])
                      / IMAGENET_DEFAULT_STD[2 - i];
    }
    cv::merge(channels, image);

    return image;
}


cv::Mat dino_preprocess(cv::Mat &img, const cv::Size img_size, const dino_hparams &params) {
    // 1) Convert to float and resize
    cv::Mat image;
    img.convertTo(image, CV_32FC3, 1.0 / 255.0);

    const auto new_size = cv::Size((image.cols / params.patch_size + 1) * params.patch_size,
                                   (image.rows / params.patch_size + 1) * params.patch_size);
    cv::resize(image, image,
               new_size,
               0, 0, cv::INTER_CUBIC);

    // 3) Channel-wise standardization
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - IMAGENET_DEFAULT_MEAN[2 - i])
                      / IMAGENET_DEFAULT_STD[2 - i];
    }
    cv::merge(channels, image);

    return image;
}


std::vector<float> interpolate_pos_embed(
    const cv::Size img_size,
    const float *pos_embed_data, // Input data shouldn't be modified
    const dino_hparams &hparams) {
    // --- Calculate New Grid Dimensions ---
    const int h_new = img_size.height / hparams.patch_size;
    const int w_new = img_size.width / hparams.patch_size;
    const int num_patches_new = h_new * w_new;

    // --- Calculate Original Grid Dimensions ---
    const int M = hparams.n_img_embd(); // Original grid side length
    const int h_orig = M;
    const int w_orig = M;
    const int num_patches_orig = h_orig * w_orig; // N = M*M
    const int hidden_sz = hparams.hidden_size; // Alias for clarity

    // --- Early Return Check ---
    if (num_patches_new == num_patches_orig) {
        const size_t total_elements = (size_t) (num_patches_orig + 1) * hidden_sz;
        return {pos_embed_data, pos_embed_data + total_elements};
    }

    // --- Prepare Output Vector ---
    const size_t total_elements_new = (size_t) (num_patches_new + 1) * hidden_sz;
    std::vector<float> pos_embed_new(total_elements_new);

    // --- Step 1: Copy CLS token embedding directly ---
    // The first hidden_sz elements are the CLS token.
    std::copy(pos_embed_data,
              pos_embed_data + hidden_sz,
              pos_embed_new.data());

    // --- Step 2: Interpolate Patch Embeddings (Dimension by Dimension) ---
    // Although data is [N, H], we process H slices of [N] shaped spatially.
    for (int c = 0; c < hidden_sz; ++c) {
        // Create a 2D grid for the *original* patches for the current hidden dimension 'c'.
        cv::Mat src_grid(h_orig, w_orig, CV_32F);

        // Gather data for the c-th dimension from all original patches.
        for (int i = 0; i < num_patches_orig; ++i) {
            const int y_orig = i / w_orig;
            const int x_orig = i % w_orig;

            // Index for the c-th component of the i-th patch embedding.
            // (i+1) because the first "row" (index 0) is the CLS token.
            size_t input_idx = (size_t) (i + 1) * hidden_sz + c;
            src_grid.at<float>(y_orig, x_orig) = pos_embed_data[input_idx];
        }

        // Resize the 2D grid for the current dimension.
        cv::Mat dst_grid;
        cv::resize(src_grid, dst_grid, cv::Size(w_new, h_new), 0, 0, cv::INTER_CUBIC);

        // Scatter the interpolated data back into the new embedding vector.
        for (int i = 0; i < num_patches_new; ++i) {
            const int y_new = i / w_new;
            const int x_new = i % w_new;

            // Index for the c-th component of the i-th *new* patch embedding.
            // (i+1) because the first "row" (index 0) is the CLS token.
            size_t output_idx = (size_t) (i + 1) * hidden_sz + c;
            pos_embed_new[output_idx] = dst_grid.at<float>(y_new, x_new);
        }
    }

    return pos_embed_new;
}

bool do_quantize(const char *name, const struct ggml_tensor *tensor) {
    bool quantize = false;
    if (std::regex_match(name, std::regex(PATTERN)))
        quantize = true;

    // quantize only 2D tensors
    quantize &= (ggml_n_dims(tensor) == 2);

    return quantize;
}

// load the model's weights from a file following the ggml format(gguf)
bool dino_model_load(const cv::Size img_size, const std::string &fname, dino_model &model, const dino_params &params) {
    printf("%s: loading model from '%s' - please wait\n", __func__, fname.c_str());
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    struct ggml_context *tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    // load hparams
    // override defaults
    auto &hparams = model.hparams;
    hparams.hidden_size = get_val_u32(gguf_ctx, std::string("hidden_size").c_str());
    hparams.num_hidden_layers = get_val_u32(gguf_ctx, std::string("num_hidden_layers").c_str());
    hparams.num_attention_heads = get_val_u32(gguf_ctx, std::string("num_attention_heads").c_str());

    hparams.patch_size = get_val_u32(gguf_ctx, std::string("patch_size").c_str());
    hparams.img_size = get_val_u32(gguf_ctx, std::string("img_size").c_str());
    hparams.ftype = get_val_u32(gguf_ctx, std::string("ftype").c_str());
    hparams.num_register_tokens = get_val_u32(gguf_ctx, std::string("num_register_tokens").c_str());

    const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

    printf("%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
    printf("%s: num_hidden_layers      = %d\n", __func__, hparams.num_hidden_layers);
    printf("%s: num_register_tokens    = %d\n", __func__, hparams.num_register_tokens);
    printf("%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
    printf("%s: patch_size             = %d\n", __func__, hparams.patch_size);
    printf("%s: img_size               = %d\n", __func__, hparams.img_size);
    printf("%s: ftype                  = %d\n", __func__, hparams.ftype);
    printf("%s: qntvr                  = %d\n", __func__, qntvr);

    if (params.classify) {
        hparams.num_classes = get_val_u32(gguf_ctx, std::string("num_classes").c_str());
        printf("%s: num_classes            = %d\n", __func__, hparams.num_classes);
        // read id2label dictionary into an ordered map (sort of an OrderedDict)
        int num_labels = get_val_u32(gguf_ctx, std::string("num_classes").c_str());
        for (int i = 0; i < num_labels; ++i) {
            model.hparams.id2label[i] = get_val_str(gguf_ctx, std::to_string(i).c_str());
        }
    }

    hparams.ftype %= GGML_QNT_VERSION_FACTOR;

    int num_tensors = gguf_get_n_tensors(gguf_ctx) + 1; // +1 for new_pos_embed

    // std::cout << "patch size " << hparams.patch_size << std::endl;

    const auto new_size = cv::Size((img_size.width / model.hparams.patch_size + 1) * model.hparams.patch_size,
                                   (img_size.height / model.hparams.patch_size + 1) * model.hparams.patch_size);

    const int h0 = new_size.height / hparams.patch_size;
    const int w0 = new_size.width / hparams.patch_size;
    const int num_patches = h0 * w0;
    const int model_num_patches = hparams.n_img_embd() * hparams.n_img_embd();

    const int offset = std::max(num_patches - model_num_patches, 0);

    struct ggml_init_params model_params{
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors + offset,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(model_params);
    for (int i = 0; i < num_tensors - 1; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor *src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor *dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
        model.tensors[name] = dst;
        std::cout << "i: " << i << ", name: " << name << ", type: " << ggml_type_name(dst->type) << std::endl;
    }


    gguf_free(gguf_ctx);

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor *cur = ggml_get_first_tensor(model.ctx); cur != nullptr;
         cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor *src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }


    return true;
}


bool dino_model_quantize(const std::string &fname_inp,
                         const std::string &fname_out,
                         int itype) {
    const auto quant_type = static_cast<ggml_type>(itype);

    struct ggml_context *tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(
        fname_inp.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n",
                __func__);
        return false;
    }

    const int num_tensors = gguf_get_n_tensors(gguf_ctx);

    gguf_context *gguf_save = gguf_init_empty();
    gguf_set_kv(gguf_save, gguf_ctx);
    gguf_set_val_u32(gguf_save, "ftype", itype);

    std::vector<std::vector<uint8_t> > buffers(num_tensors);
    ggml_type new_type;
    bool do_q = false;

    for (int i = 0; i < num_tensors; i++) {
        const char *name =
                gguf_get_tensor_name(gguf_ctx, i);
        const struct ggml_tensor *tensor =
                ggml_get_tensor(tmp_ctx, name);
        gguf_add_tensor(gguf_save, tensor);

        auto &work_bytes = buffers[i];
        const size_t byte_size = ggml_nbytes(tensor);
        work_bytes.resize(byte_size);
        void *new_data = work_bytes.data();
        size_t new_size = 0;

        do_q = do_quantize(name, tensor);

        if (do_q) {
            new_type = quant_type;
            const bool is_fp16 = tensor->type == GGML_TYPE_F16;
            const float *data_f32 = nullptr;
            if (is_fp16) {
                std::vector<float> f16_to_f32;
                const int64_t ne = ggml_nelements(tensor);
                f16_to_f32.resize(ne);
                const uint16_t *src16 = static_cast<uint16_t *>(tensor->data);
                for (int64_t j = 0; j < ne; ++j) {
                    f16_to_f32[j] = ggml_fp16_to_fp32(src16[j]);
                }
                data_f32 = f16_to_f32.data();
            } else {
                data_f32 = ggml_get_data_f32(tensor);
            }
            new_size = ggml_quantize_chunk(
                quant_type,
                data_f32,
                new_data,
                0,
                tensor->ne[1],
                tensor->ne[0],
                nullptr
            );
            if (!ggml_validate_row_data(
                quant_type, new_data, new_size)) {
                throw std::runtime_error(
                    "quantized data validation failed");
            }
        } else {
            new_type = tensor->type;
            memcpy(new_data, tensor->data, byte_size);
            new_size = byte_size;
        }

        gguf_set_tensor_type(gguf_save, name, new_type);
        GGML_ASSERT(
            gguf_get_tensor_size(
                gguf_save,
                gguf_find_tensor(gguf_save, name))
            == new_size);
        gguf_set_tensor_data(
            gguf_save, name, new_data);
    }

    if (!gguf_write_to_file(
        gguf_save, fname_out.c_str(), false)) {
        fprintf(stderr,
                "failed to write GGUF file\n");
    }

    gguf_free(gguf_ctx);
    gguf_free(gguf_save);
    return true;
}


// DINOv2 Encoder

struct ggml_tensor *attn(struct ggml_tensor *cur, const float scale, const int il, struct ggml_context *ctx_cgraph,
                         const dino_model &model, const dino_params &params) {
    const uint32_t num_attention_heads = model.hparams.num_attention_heads;
    const uint32_t n_enc_head_dim = model.hparams.n_enc_head_dim();
    const uint32_t hidden_size = model.hparams.hidden_size;
    const int64_t W = cur->ne[1];
    const int64_t H = cur->ne[2];
    const int64_t total_patches = W * H;

    // self-attention

    cur = ggml_mul_mat(
        ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.attention.qkv.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.attention.qkv.bias"));


    // split qkv into separate tensors
    const int B = cur->ne[3];

    cur = ggml_reshape_4d(ctx_cgraph, cur, hidden_size, 3, W * H, B);
    cur = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, cur, 0, 3, 1, 2));

    struct ggml_tensor *Q = ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2],
                                         0 * cur->nb[3]);
    Q = ggml_reshape_4d(ctx_cgraph, Q, n_enc_head_dim, num_attention_heads, W * H, B);
    Q = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, Q, 0, 2, 1, 3));

    struct ggml_tensor *K = ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2],
                                         1 * cur->nb[3]);
    K = ggml_reshape_4d(ctx_cgraph, K, n_enc_head_dim, num_attention_heads, W * H, B);
    K = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, K, 0, 2, 1, 3));

    struct ggml_tensor *V = ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2],
                                         2 * cur->nb[3]);
    V = ggml_reshape_4d(ctx_cgraph, V, n_enc_head_dim, num_attention_heads, W * H, B);

    // std::cout << "K type " << ggml_type_name(K->type) << std::endl;


    if (params.enable_flash_attn) {
        const int64_t total_patches_padding = GGML_PAD(total_patches, 32);
        const int64_t total_patches_to_pad = total_patches_padding - total_patches;

        const int64_t hidden_size_padding = GGML_PAD(hidden_size, 4);
        const int64_t hidden_size_to_pad = hidden_size_padding - hidden_size;

        V = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 0, 2, 1, 3));

        Q = ggml_pad(ctx_cgraph, Q, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        K = ggml_pad(ctx_cgraph, K, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        V = ggml_pad(ctx_cgraph, V, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        // K = ggml_cpy(ctx_cgraph, K,
        //              ggml_new_tensor_4d(ctx_cgraph, cur->type, n_enc_head_dim, total_patches_padding,
        //                                 num_attention_heads,
        //                                 1));
        //
        // V = ggml_cpy(ctx_cgraph, V,
        //              ggml_new_tensor_4d(ctx_cgraph, cur->type, n_enc_head_dim, total_patches_padding,
        //                                 num_attention_heads,
        //                                 1));

        struct ggml_tensor *KQV = ggml_flash_attn_ext(ctx_cgraph, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        KQV = ggml_view_4d(ctx_cgraph, KQV, KQV->ne[0], KQV->ne[1], KQV->ne[2] - total_patches_to_pad, KQV->ne[3],
                           KQV->nb[1], KQV->nb[2], KQV->nb[3], 0);

        cur = ggml_reshape_4d(ctx_cgraph,
                              KQV,
                              hidden_size, W, H, 1);
    } else {
        Q = ggml_reshape_3d(ctx_cgraph, Q, n_enc_head_dim, W * H, B * num_attention_heads);
        K = ggml_reshape_3d(ctx_cgraph, K, n_enc_head_dim, W * H, B * num_attention_heads);
        V = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 1, 2, 0, 3)); // transposed
        V = ggml_reshape_3d(ctx_cgraph, V, W * H, n_enc_head_dim, B * num_attention_heads);
        struct ggml_tensor *KQ = ggml_mul_mat(ctx_cgraph, K, Q);

        // attention weights
        struct ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx_cgraph, KQ, nullptr, scale, 0.0f);

        struct ggml_tensor *KQV = ggml_mul_mat(ctx_cgraph, V, KQ_soft_max);

        cur = ggml_reshape_4d(ctx_cgraph,
                              ggml_cont(ctx_cgraph,
                                        ggml_permute(ctx_cgraph,
                                                     KQV,
                                                     0, 2, 1, 3)),
                              hidden_size, W, H, 1);
    }

    cur = ggml_mul_mat(
        ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.output.dense.weight"),
        cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at(
                               "encoder.layer." + std::to_string(il) + ".attention.output.dense.bias"));

    return cur;
}

struct ggml_tensor *mlp(struct ggml_tensor *cur, const int il, struct ggml_context *ctx_cgraph,
                        const dino_model &model,
                        const dino_params &params) {
    // fully connected layer
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight"),
                       cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.bias"));

    // GELU activation
    cur = ggml_gelu(ctx_cgraph, cur);

    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;

    // projection
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc2.weight"),
                       cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc2.bias"));
    return cur;
}

struct ggml_tensor *swiglu_ffn(struct ggml_tensor *cur, const int il, struct ggml_context *ctx_cgraph,
                               const dino_model &model,
                               const dino_params &params) {
    // fully connected layer
    cur = ggml_mul_mat(
        ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.weights_in.weight"),
        cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.weights_in.bias"));

    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;

    int64_t ne0 = cur->ne[0] / 2;
    int64_t ne1 = cur->ne[1];
    int64_t ne2 = cur->ne[2];
    int64_t ne3 = cur->ne[3];
    size_t nb0 = cur->nb[0];
    size_t nb1 = cur->nb[1];
    size_t nb2 = cur->nb[2];
    size_t nb3 = cur->nb[3];
    size_t offset = nb0 * ne0;

    struct ggml_tensor *cur1 = ggml_view_4d(ctx_cgraph, cur, ne0, ne1, ne2, ne3,
                                            nb1, nb2, nb3, 0);

    struct ggml_tensor *cur2 = ggml_view_4d(ctx_cgraph, cur, ne0, ne1, ne2, ne3,
                                            nb1, nb2, nb3, offset);

    // SILU activation
    cur = ggml_mul_inplace(ctx_cgraph, ggml_silu(ctx_cgraph, ggml_cont(ctx_cgraph, cur1)), cur2);

    // projection
    cur = ggml_mul_mat(
        ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.weights_out.weight"),
        cur);
    cur = ggml_add_inplace(ctx_cgraph, cur,
                           model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.weights_out.bias"));
    return cur;
}

void forward_features(const cv::Size img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                      const dino_model &model, const dino_params &params) {
    const uint32_t hidden_size = model.hparams.hidden_size;
    const uint32_t num_hidden_layers = model.hparams.num_hidden_layers;
    const uint32_t n_enc_head_dim = model.hparams.n_enc_head_dim();
    const uint32_t num_register_tokens = model.hparams.num_register_tokens;
    const int h0 = img_size.height / model.hparams.patch_size;
    const int w0 = img_size.width / model.hparams.patch_size;
    const int num_patches = h0 * w0;

    const float scale = 1.0f / sqrtf(static_cast<float>(n_enc_head_dim));
    // (W, H, C, B)
    // (518, 518, 3, 1)
    struct ggml_tensor *input =
            ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, img_size.width, img_size.height, 3, 1);
    ggml_set_name(input, "input");

    // patch embedding
    // (37, 37, 768, 1)
    // std::cout << "patch embed " << enc.patch_embed_w->ne[0] << std::endl;
    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(
        ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.weight"), input);


    cur = ggml_add_inplace(ctx_cgraph,
                           ggml_repeat(ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.bias"),
                                       cur), cur); // (37, 37, 768, 1)


    cur = ggml_cont(ctx_cgraph,
                    ggml_permute(ctx_cgraph, cur, 1, 2, 0, 3)); // (37, 768, 37, 1)
    //
    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;

    //
    // add positional embedding
    // cur dim     : 768  37  37  1
    // enc.pe dim  : 768  1370  1  1

    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;
    //
    // reshape patch embeddings from (768  37  37  1) to (768  1369  1  1)
    cur = ggml_reshape_4d(ctx_cgraph, cur, hidden_size, num_patches, 1, 1);

    struct ggml_tensor *pos_embed_fixed = ggml_new_tensor_3d(
        ctx_cgraph, model.tensors.at("embeddings.position_embeddings")->type, model.hparams.hidden_size,
        num_patches + 1, 1
    );

    ggml_set_name(pos_embed_fixed, "pos_embed_fixed");

    cur = ggml_concat(ctx_cgraph, model.tensors.at("embeddings.cls_token"), cur, 1);

    cur = ggml_add_inplace(ctx_cgraph, cur, pos_embed_fixed);

    if (num_register_tokens > 0) {
        struct ggml_tensor *cls_token = ggml_view_1d(ctx_cgraph, cur, hidden_size, 0);
        struct ggml_tensor *patch_tokens = ggml_view_4d(ctx_cgraph, cur, cur->ne[0], cur->ne[1] - 1,
                                                        cur->ne[2],
                                                        cur->ne[3],
                                                        cur->nb[1],
                                                        cur->nb[2],
                                                        cur->nb[3],
                                                        cur->nb[1]);
        cur = ggml_concat(ctx_cgraph, ggml_concat(ctx_cgraph, cls_token,
                                                  model.tensors.at("embeddings.register_tokens"),
                                                  1), patch_tokens, 1);
    }

    struct ggml_tensor *inpL = cur;
    //
    // loop over layers
    for (int il = 0; il < num_hidden_layers; ++il) {
        // norm 1
        {
            cur = ggml_norm(ctx_cgraph, inpL, model.hparams.eps);

            // cur = w * cur + b
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.weight"));
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.bias"));
        }

        // std::cout << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] << std::endl;

        // self attn
        cur = attn(cur, scale, il, ctx_cgraph, model, params);

        cur = ggml_mul_inplace(ctx_cgraph, cur,
                               model.tensors.at(
                                   "encoder.layer." + std::to_string(il) +
                                   ".layer_scale1.lambda1"));

        // add skip connection
        cur = ggml_add_inplace(ctx_cgraph, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm 2
            {
                cur = ggml_norm(ctx_cgraph, inpFF, model.hparams.eps);

                // cur = w * cur + b
                cur = ggml_mul_inplace(ctx_cgraph, cur,
                                       model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.weight"));
                cur = ggml_add_inplace(ctx_cgraph, cur,
                                       model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.bias"));
            }

            // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
            //         << std::endl;
            //
            // std::cout << "mlp.fc1 size " << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")
            //         ->ne[0] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[1] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[2] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[3] << std::endl;

            if (model.hparams.num_hidden_layers == 40)
                cur = swiglu_ffn(cur, il, ctx_cgraph, model, params);
            else
                cur = mlp(cur, il, ctx_cgraph, model, params);
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.
                                   at("encoder.layer." + std::to_string(il) + ".layer_scale2.lambda1"));
        }
        //
        inpL = ggml_add_inplace(ctx_cgraph, cur, inpFF);
    }

    cur = inpL;

    // layer normalization
    {
        cur = ggml_norm_inplace(ctx_cgraph, cur, model.hparams.eps);

        // cur = w * cur + b
        cur = ggml_mul_inplace(ctx_cgraph, cur, model.tensors.at("layernorm.weight"));
        cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("layernorm.bias"));
    }

    // get the output of cls token at index 0
    struct ggml_tensor *cls_token = ggml_view_1d(ctx_cgraph, cur, hidden_size, 0);

    ggml_set_output(cls_token);
    ggml_set_name(cls_token, "cls_token");
    ggml_build_forward_expand(graph, cls_token);

    int64_t ne1 = cur->ne[1] - 1;
    size_t offset = cur->nb[1];
    if (!params.classify) {
        // include register tokens for classification pooling
        ne1 -= num_register_tokens;
        offset *= (num_register_tokens + 1);
    }

    struct ggml_tensor *patch_tokens = ggml_view_4d(ctx_cgraph, cur, cur->ne[0],
                                                    ne1,
                                                    cur->ne[2],
                                                    cur->ne[3],
                                                    cur->nb[1],
                                                    cur->nb[2],
                                                    cur->nb[3],
                                                    offset);

    ggml_set_output(patch_tokens);
    ggml_set_name(patch_tokens, "patch_tokens");
    ggml_build_forward_expand(graph, patch_tokens);
}

void forward_head(const cv::Size img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                  const dino_model &model, const dino_params &params) {
    const int32_t n_img_embd = model.hparams.n_img_embd();

    struct ggml_tensor *cls_token = ggml_graph_get_tensor(graph, "cls_token");
    struct ggml_tensor *patch_tokens = ggml_graph_get_tensor(graph, "patch_tokens");
    // classification head

    struct ggml_tensor *pooled_patch_tokens = ggml_sum_rows(
        ctx_cgraph, ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, patch_tokens, 1, 0, 2, 3)));
    pooled_patch_tokens = ggml_scale_inplace(ctx_cgraph, pooled_patch_tokens,
                                             1.0f / static_cast<float>(n_img_embd * n_img_embd));

    struct ggml_tensor *cur = ggml_concat(ctx_cgraph, cls_token, ggml_permute(
                                              ctx_cgraph, pooled_patch_tokens, 1,
                                              0, 2,
                                              3), 0);

    // projection
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("classifier.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("classifier.bias"));

    // softmax
    ggml_tensor *probs = ggml_soft_max(ctx_cgraph, cur);
    //
    ggml_set_output(probs);
    ggml_set_name(probs, "probs");

    ggml_build_forward_expand(graph, probs);
}

struct ggml_cgraph *build_graph(
    const cv::Size img_size,
    struct ggml_context *ctx_cgraph,
    const dino_model &model,
    const dino_params &params) {
    const auto &hparams = model.hparams;

    struct ggml_cgraph *gf = ggml_new_graph(ctx_cgraph);

    forward_features(img_size, gf, ctx_cgraph, model, params);

    if (params.classify)
        forward_head(img_size, gf, ctx_cgraph, model, params);

    return gf;
}

void print_usage(int argc, char **argv, const dino_params &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help              show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model       model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp         input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out         output file for backbone PCA features (default: %s)\n",
            params.image_out.c_str());
    fprintf(stderr, "  -k N, --topk            top k classes to print (default: %d)\n", params.topk);
    fprintf(stderr, "  -t N, --threads         number of threads to use during computation (default: %d)\n",
            params.n_threads);
    fprintf(
        stderr, "  -c, --classify          whether to classify the image or get backbone PCA features (default: %d)\n",
        params.classify);
    fprintf(
        stderr, "  -fa, --flash_attn          whether to enable flash_attn, less accurate (default: %d)\n",
        params.enable_flash_attn);
    fprintf(
        stderr,
        "  -cid, --camera_id          the idea of the camera for realtime backbone PCA feature streaming (default: %d)\n",
        params.camera_id);
    fprintf(stderr, "\n");
}

bool dino_params_parse(int argc, char **argv, dino_params &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_inp = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-k" || arg == "--topk") {
            params.topk = std::stoi(argv[++i]);
        } else if (arg == "-cid" || arg == "--camera_id") {
            params.camera_id = std::stoi(argv[++i]);
        } else if (arg == "-fa" || arg == "--flash_attn") {
            params.enable_flash_attn = true;
        } else if (arg == "-c" || arg == "--classify") {
            params.classify = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

std::unique_ptr<dino_output> dino_predict(const dino_model &model, const cv::Mat &img,
                                          const dino_params &params) {
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context *ctx_cgraph = ggml_init(params0);
    struct ggml_cgraph *gf = build_graph(img.size(), ctx_cgraph, model, params);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    struct ggml_tensor *input = ggml_graph_get_tensor(gf, "input");

    std::vector<float> planar(img.total() * 3);
    float *out = planar.data();

    // Split BGR image into channels (OpenCV will do B, G, R)
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(img, bgr_channels);

    // Create output Mats for planar format in **RGB order**
    std::vector<cv::Mat> rgb_planar_channels = {
        cv::Mat(img.rows, img.cols, CV_32F, out + 0 * img.total()), // R
        cv::Mat(img.rows, img.cols, CV_32F, out + 1 * img.total()), // G
        cv::Mat(img.rows, img.cols, CV_32F, out + 2 * img.total()) // B
    };

    // Copy from BGR channels into RGB-planar layout
    bgr_channels[2].copyTo(rgb_planar_channels[0]); // R <- from BGR[2]
    bgr_channels[1].copyTo(rgb_planar_channels[1]); // G <- from BGR[1]
    bgr_channels[0].copyTo(rgb_planar_channels[2]); // B <- from BGR[0]

    ggml_backend_tensor_set(input, planar.data(), 0, ggml_nbytes(input));

    const struct ggml_tensor *pos_embed = ggml_get_tensor(model.ctx, "embeddings.position_embeddings");

    const std::vector<float> pos_embed_fixed_data = interpolate_pos_embed(
        img.size(), ggml_get_data_f32(pos_embed), model.hparams);

    struct ggml_tensor *pos_embed_fixed = ggml_graph_get_tensor(gf, "pos_embed_fixed");

    ggml_backend_tensor_set(pos_embed_fixed, pos_embed_fixed_data.data(), 0, ggml_nbytes(pos_embed_fixed));
    // print_t_f32("pos_embed_fixed", pos_embed_fixed);

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return {};
    }

    auto output = std::make_unique<dino_output>();

    if (params.classify) {
        struct ggml_tensor *probs = ggml_graph_get_tensor(gf, "probs");
        const float *probs_data = ggml_get_data_f32(probs);
        std::vector<std::pair<float, int> > predictions;
        // store probability and index
        for (int i = 0; i < model.hparams.num_classes; ++i) {
            predictions.emplace_back(probs_data[i], i);
        }

        // sort in descending order
        std::sort(predictions.begin(), predictions.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                      return a.first > b.first;
                  });

        fprintf(stderr, "\n");

        // top k predictions
        std::vector<uint32_t> preds(params.topk);
        for (int i = 0; i < params.topk && i < predictions.size(); ++i) {
            printf(" > %s : %.2f\n",
                   model.hparams.id2label.at(predictions[i].second).c_str(),
                   predictions[i].first);
            preds[i] = static_cast<uint32_t>(predictions[i].first);
        }

        output->preds = preds;
    } else {
        struct ggml_tensor *patches = ggml_graph_get_tensor(gf, "patch_tokens");
        const float *patch_tokens_data = ggml_get_data_f32(patches);
        const int h0 = img.rows / model.hparams.patch_size;
        const int w0 = img.cols / model.hparams.patch_size;
        const int num_patches = h0 * w0;
        // Allocate cv::Mat (which allocates and owns memory)
        cv::Mat patch_tokens(num_patches, model.hparams.hidden_size, CV_32F);

        // Copy data from ggml tensor into cv::Mat
        std::memcpy(patch_tokens.data, patch_tokens_data, num_patches * model.hparams.hidden_size * sizeof(float));

        // Store in your output struct
        output->patch_tokens = patch_tokens;
    }


    // free memory
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);


    return output;
}



