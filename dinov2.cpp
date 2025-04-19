#define CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "dinov2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ggml/examples/stb_image.h" // stb image load
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
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

int32_t dino_hparams::n_enc_head_dim() const {
    return hidden_size / num_attention_heads;
}

int32_t dino_hparams::n_img_size() const {
    return img_size;
}

int32_t dino_hparams::n_patch_size() const {
    return patch_size;
}

int32_t dino_hparams::n_img_embd() const {
    return n_img_size() / n_patch_size();
}

u_int32_t get_val_u32(const struct gguf_context *ctx,
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

// load image from a file(uses stbi_load)
bool load_image_from_file(const std::string &fname, image_u8 &img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

// preprocess input image : bilinear resize + normalize
bool dino_image_preprocess_bilinear(const image_u8 &img, image_f32 &res, const dino_hparams &params) {
    const int nx = img.nx;
    const int ny = img.ny;

    const int target_size = params.n_img_size();

    res.nx = target_size;
    res.ny = target_size;
    res.data.resize(3 * target_size * target_size);

    const float x_scale = nx / (float) target_size;
    const float y_scale = ny / (float) target_size;

    fprintf(stderr, "%s: x_scale = %f, y_scale = %f\n", __func__, x_scale, y_scale);

    const int nx3 = int(nx / x_scale + 0.5f);
    const int ny3 = int(ny / y_scale + 0.5f);

    const float m3[3] = {123.675f, 116.280f, 103.530f};
    const float s3[3] = {58.395f, 57.120f, 57.375f};

    // #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * x_scale - 0.5f;
                const float sy = (y + 0.5f) * y_scale - 0.5f;

                const int x0 = std::max(0, (int) std::floor(sx));
                const int y0 = std::max(0, (int) std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = img.data[j00];
                const float v01 = img.data[j01];
                const float v10 = img.data[j10];
                const float v11 = img.data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res.data[i] = (float(v2) - m3[c]) / s3[c];
            }
        }
    }
    return true;
}

float clip(float x, float lower, float upper) {
    return std::max(lower, std::min(x, upper));
}

// preprocess input image : bicubic resize + normalize
bool dino_image_preprocess_bicubic(const image_u8 &img, image_f32 &res, const dino_hparams &params) {
    const int nx = img.nx;
    const int ny = img.ny;

    const int newWidth = params.n_img_size();
    const int newHeight = params.n_img_size();
    res.nx = newWidth;
    res.ny = newHeight;
    res.data.resize(3 * newWidth * newHeight);

    int a, b, c, d, index;
    float Ca, Cb, Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, ii, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float) nx / (float) newWidth;
    ty = (float) ny / (float) newHeight;
    printf("newWidth, newHeight = %d, %d\n", newWidth, newHeight);
    printf("tx, ty = %f, %f\n", tx, ty);
    printf("nx, ny = %d, %d\n", nx, ny);

    float scale = std::max(tx, ty);
    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const float m3[3] = {123.675f, 116.280f, 103.530f};
    const float s3[3] = {58.395f, 57.120f, 57.375f};

    // Bicubic interpolation; inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    // #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < newHeight; i++) {
        for (j = 0; j < newWidth; j++) {
            x = (int) (tx * j);
            y = (int) (ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            index = (y * nx + x) * 3;
            a = (y * nx + (x + 1)) * 3;
            b = ((y + 1) * nx + x) * 3;
            c = ((y + 1) * nx + (x + 1)) * 3;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.data[
                             (clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.data[
                             (clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.data[
                             (clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    res.data[(i * newWidth + j) * 3 + k] = (float(Cc2) - m3[k]) / s3[k];
                }
            }
        }
    }

    return true;
}

bool dino_image_preprocess(const image_u8 &img, image_f32 &res, const dino_hparams &params) {
    const std::string mode = params.interpolation.c_str();
    if (mode == "bilinear") {
        return dino_image_preprocess_bilinear(img, res, params);
    } else if (mode == "bicubic") {
        return dino_image_preprocess_bicubic(img, res, params);
    } else {
        std::cout << "Interpolation mode '" << mode << "' is not supported; returning 'false'...";
        return false;
    }
}

// load the model's weights from a file following the ggml format(gguf)
bool dino_model_load(const std::string &fname, dino_model &model, const dino_params &params) {
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

    int num_tensors = gguf_get_n_tensors(gguf_ctx);

    struct ggml_init_params model_params{
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(model_params);
    for (int i = 0; i < num_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor *src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor *dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
        model.tensors[name] = dst;
    }
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor *cur = ggml_get_first_tensor(model.ctx); cur != nullptr;
         cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor *src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }


    // load hparams
    {
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
    }

    gguf_free(gguf_ctx);

    return true;
}

// DINOv2 Encoder

void *forward_features(struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                       const dino_model &model, const dino_params &params) {
    const int32_t hidden_size = model.hparams.hidden_size;
    const int32_t num_hidden_layers = model.hparams.num_hidden_layers;
    const int32_t num_attention_heads = model.hparams.num_attention_heads;
    const int32_t n_img_size = model.hparams.img_size;
    const int32_t n_enc_head_dim = model.hparams.n_enc_head_dim();
    const int32_t n_img_embd = model.hparams.n_img_embd();
    const int32_t num_register_tokens = model.hparams.num_register_tokens;
    // (W, H, C, B)
    // (518, 518, 3, 1)
    struct ggml_tensor *input = ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    ggml_set_name(input, "input");

    // patch embedding
    // (37, 37, 768, 1)
    // std::cout << "patch embed " << enc.patch_embed_w->ne[0] << std::endl;
    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(
        ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.weight"), input);
    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] <<
    //         std::endl;
    // std::cout << "enc patch embed shape " << enc.patch_embed_w->ne[0] << ", " << enc.patch_embed_w->ne[1] << ", " << enc
    //         .patch_embed_w->
    //         ne[2] << ", " << enc.patch_embed_w->ne[3] << std::endl;
    cur = ggml_add_inplace(ctx_cgraph,
                           cur,
                           ggml_repeat(ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.bias"),
                                       cur)); // (37, 37, 768, 1)

    // std::cout << "shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] << std::endl;
    // (37, 37, 768, 1)

    cur = ggml_cont(ctx_cgraph,
                    ggml_permute(ctx_cgraph, cur, 1, 2, 0, 3)); // (37, 768, 37, 1)
    //
    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;

    //
    // add positional embedding
    // cur dim     : 768  37  37  1
    // enc.pe dim  : 768  1370  1  1
    //
    // reshape patch embeddings from (768  37  37  1) to (768  1369  1  1)
    cur = ggml_reshape_4d(ctx_cgraph, cur, hidden_size, n_img_embd * n_img_embd, 1, 1);

    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;
    //
    // concat class embeddings(cls_token) : (768  1  1  1) with positional embeddings (pos_embed = cur) : (768  1369  1  1)
    //
    // std::cout << "cls_token shape " << cur2->ne[0] << ", " << cur2->ne[1] << ", " << cur2->ne[2] << ", " << cur2->ne[3]
    //         << std::endl;

    // std::cout << "embeddings.register_tokens shape " << model.tensors.at("embeddings.register_tokens")->ne[0] << ", "
    //         << model.tensors.at("embeddings.register_tokens")->ne[1] << ", "
    //         << model.tensors.at("embeddings.register_tokens")->ne[2] << ", "
    //         << model.tensors.at("embeddings.register_tokens")->ne[3] << std::endl;

    cur = ggml_concat(ctx_cgraph, model.tensors.at("embeddings.cls_token"), cur, 1);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("embeddings.position_embeddings"));

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


    // cur = ggml_permute(ctx_cgraph, cur,
    //                    0, 2, 1, 3); // 768  1370  1  1

    // std::cout << "cur shape" << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;
    //
    // std::cout << "pos embed shape" << model.tensors.at("embeddings.embeddings.projection.weight")->ne[0];


    struct ggml_tensor *inpL = cur;
    //
    // loop over layers
    for (int il = 0; il < num_hidden_layers; ++il) {
        // norm 1
        {
            cur = ggml_norm(ctx_cgraph, inpL, model.hparams.eps);

            // cur = w * cur + b
            cur = ggml_mul(ctx_cgraph, cur, model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.weight"));
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.bias"));
        }

        // std::cout << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] << std::endl;

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            struct ggml_tensor *Q;
            struct ggml_tensor *K;
            struct ggml_tensor *V;


            Q = ggml_mul_mat(
                ctx_cgraph,
                model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.attention.query.weight"),
                cur); // 768, 1370, 1, 1
            Q = ggml_add_inplace(ctx_cgraph, Q,
                                 model.tensors.at(
                                     "encoder.layer." + std::to_string(il) + ".attention.attention.query.bias"));
            Q = ggml_reshape_4d(ctx_cgraph, Q, n_enc_head_dim, num_attention_heads, W * H, 1);
            Q = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx_cgraph, Q, n_enc_head_dim, W * H, num_attention_heads);

            K = ggml_mul_mat(
                ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.attention.key.weight"),
                cur); // 768, 1370, 1, 1
            K = ggml_add_inplace(ctx_cgraph, K,
                                 model.tensors.at(
                                     "encoder.layer." + std::to_string(il) + ".attention.attention.key.bias"));
            K = ggml_reshape_4d(ctx_cgraph, K, n_enc_head_dim, num_attention_heads, W * H, 1);
            K = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx_cgraph, K, n_enc_head_dim, W * H, num_attention_heads);

            V = ggml_mul_mat(
                ctx_cgraph,
                model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.attention.value.weight"),
                cur); // 768, 1370, 1, 1
            V = ggml_add_inplace(ctx_cgraph, V,
                                 model.tensors.at(
                                     "encoder.layer." + std::to_string(il) + ".attention.attention.value.bias"));
            V = ggml_reshape_4d(ctx_cgraph, V, n_enc_head_dim, num_attention_heads, W * H, 1);
            V = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(ctx_cgraph, V, W * H, n_enc_head_dim, num_attention_heads);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx_cgraph, K, Q);

            // attention weights
            struct ggml_tensor *KQ_scaled =
                    ggml_scale_inplace(ctx_cgraph,
                                       KQ,
                                       1.0f / sqrtf(n_enc_head_dim));

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx_cgraph, KQ_scaled);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx_cgraph, V, KQ_soft_max);

            cur =
                    ggml_reshape_4d(ctx_cgraph,
                                    ggml_cont(ctx_cgraph,
                                              ggml_permute(ctx_cgraph,
                                                           ggml_reshape_4d(
                                                               ctx_cgraph, KQV, n_enc_head_dim, W * H,
                                                               num_attention_heads,
                                                               1),
                                                           0, 2, 1, 3)),
                                    hidden_size, W, H, 1);

            cur = ggml_mul_mat(
                ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".attention.output.dense.weight"),
                cur);
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at(
                                       "encoder.layer." + std::to_string(il) + ".attention.output.dense.bias"));
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.at(
                                       "encoder.layer." + std::to_string(il) +
                                       ".layer_scale1.lambda1"));
        }

        // add skip connection
        cur = ggml_add_inplace(ctx_cgraph, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm 2
            {
                cur = ggml_norm(ctx_cgraph, inpFF, model.hparams.eps);

                // cur = w * cur + b
                cur = ggml_mul(ctx_cgraph, cur,
                               model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.weight"));
                cur = ggml_add_inplace(ctx_cgraph, cur,
                                       model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.bias"));
            }

            // fully connected layer
            cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight"),
                               cur);
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.bias"));

            // GELU activation
            cur = ggml_gelu(ctx_cgraph, cur);

            // projection
            cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc2.weight"),
                               cur);
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc2.bias"));
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.
                                   at("encoder.layer." + std::to_string(il) + ".layer_scale2.lambda1"));
        }
        //
        inpL = ggml_add(ctx_cgraph, cur, inpFF);
    }

    cur = inpL;

    //
    // pooling
    //


    // layer normalization
    {
        cur = ggml_norm(ctx_cgraph, cur, model.hparams.eps);

        // cur = w * cur + b
        cur = ggml_mul(ctx_cgraph, cur, model.tensors.at("layernorm.weight"));
        cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("layernorm.bias"));
    }

    // get the output of cls token at index 0
    struct ggml_tensor *cls_token = ggml_view_1d(ctx_cgraph, cur, hidden_size, 0);

    ggml_set_output(cls_token);
    ggml_set_name(cls_token, "cls_token");
    ggml_build_forward_expand(graph, cls_token);

    int64_t offset = cur->ne[1] - 1;
    if (!params.classify) // include register tokens for classification pooling
        offset -= num_register_tokens;

    struct ggml_tensor *patch_tokens = ggml_view_4d(ctx_cgraph, cur, cur->ne[0],
                                                    offset,
                                                    cur->ne[2],
                                                    cur->ne[3],
                                                    cur->nb[1],
                                                    cur->nb[2],
                                                    cur->nb[3],
                                                    cur->nb[1]);


    // std::cout << "patch tokens shape: " << patch_tokens->ne[0] << ", " << patch_tokens->ne[1] << ", "
    //         << patch_tokens->ne[2] << ", " << patch_tokens->ne[3] << std::endl;

    ggml_set_output(patch_tokens);
    ggml_set_name(patch_tokens, "patch_tokens");
    ggml_build_forward_expand(graph, patch_tokens);
}

void *forward_head(struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                   const dino_model &model, const dino_params &params) {
    const int32_t n_img_embd = model.hparams.n_img_embd();

    struct ggml_tensor *cls_token = ggml_graph_get_tensor(graph, "cls_token");
    struct ggml_tensor *patch_tokens = ggml_graph_get_tensor(graph, "patch_tokens");
    // classification head


    // patch_tokens = ggml_cont(ctx0, ggml_permute(ctx0, patch_tokens, 1, 0, 2, 3));

    // std::cout << "cls tokens: " << cls_token->ne[0] << ", " << cls_token->ne[1] << ", " << cls_token->ne[2] << ", " <<
    //         cls_token->ne[3]
    //         << std::endl;
    // std::cout << "patch tokens shape " << patch_tokens->ne[0] << ", " << patch_tokens->ne[1] << ", " << patch_tokens->ne
    //         [2] << ", "
    //         << patch_tokens->ne[3] << std::endl;

    struct ggml_tensor *pooled_patch_tokens = ggml_sum_rows(
        ctx_cgraph, ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, patch_tokens, 1, 0, 2, 3)));
    pooled_patch_tokens = ggml_scale_inplace(ctx_cgraph, pooled_patch_tokens, 1.0f / (n_img_embd * n_img_embd));

    struct ggml_tensor *cur = ggml_concat(ctx_cgraph, cls_token, ggml_permute(
                                              ctx_cgraph, pooled_patch_tokens, 1,
                                              0, 2,
                                              3), 0);


    // std::cout << "cls_token shape " << cls_token->ne[0] << ", " << cls_token->ne[1] << ", " << cls_token->ne[2] << ", "
    //         << cls_token->ne[3] << std::endl;
    // std::cout << "patch_tokens shape " << patch_tokens->ne[0] << ", " << patch_tokens->ne[1] << ", " << patch_tokens->ne
    //         [2]
    //         << ", " << patch_tokens->ne[3] << std::endl;
    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] <<
    //         std::endl;

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
    struct ggml_context *ctx_cgraph,
    const dino_model &model,
    const dino_params &params) {
    const auto &hparams = model.hparams;

    struct ggml_cgraph *gf = ggml_new_graph(ctx_cgraph);

    forward_features(gf, ctx_cgraph, model, params);

    if (params.classify)
        forward_head(gf, ctx_cgraph, model, params);


    return gf;
}

void print_usage(int argc, char **argv, const dino_params &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help              show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model       model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp         input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -k N, --topk            top k classes to print (default: %d)\n", params.topk);
    fprintf(stderr, "  -c, --classify          whether to classify the image or get backbone features (default: %d)\n",
            params.classify);
    fprintf(stderr, "  -s SEED, --seed         RNG seed (default: -1)\n");
    fprintf(stderr, "  -e FLOAT, --epsilon     epsilon constant in Layer Norm layers (default: %f)\n", params.eps);
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
        } else if (arg == "-k" || arg == "--topk") {
            params.topk = std::stoi(argv[++i]);
        } else if (arg == "-e" || arg == "--epsilon") {
            params.eps = std::stof(argv[++i]);
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

std::unique_ptr<dino_output> dino_predict(const dino_model &model, const image_f32 &img1,
                                          const dino_params &params) {
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context *ctx_cgraph = ggml_init(params0);
    struct ggml_cgraph *gf = build_graph(ctx_cgraph, model, params);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    struct ggml_tensor *input = ggml_graph_get_tensor(gf, "input");
    // ggml_backend_tensor_set(input, img1.data.data(), 0, ggml_nbytes(input));

    {
        float *data = (float *) ggml_get_data(input);

        const int nx = img1.nx;
        const int ny = img1.ny;
        const int n = nx * ny;

        const int32_t n_img_size = model.hparams.img_size;

        GGML_ASSERT(nx == n_img_size && ny == n_img_size);

        for (int k = 0; k < 3; k++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    data[k * n + y * nx + x] = img1.data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return {};
    }


    // print_t_f32("after probs", state.prediction);
    const int num_patches = model.hparams.n_img_embd() * model.hparams.n_img_embd();

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
            preds[i] = predictions[i].first;
        }

        output->preds = preds;
    } else {
        struct ggml_tensor *patches = ggml_graph_get_tensor(gf, "patch_tokens");
        const float *patch_tokens_data = ggml_get_data_f32(patches);
        std::vector<float> patch_tokens(patch_tokens_data,
                                        patch_tokens_data + num_patches * model.hparams.hidden_size);
        output->patch_tokens = patch_tokens;
    }


    // free memory
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);


    return output;
}

