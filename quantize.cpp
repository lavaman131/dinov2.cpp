// add simple qunatization strategies
// adapted from : ggml/gpt-2

#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>
#include <iostream>



static int find_kv_index(struct gguf_context* ctx, const char* key) {
    const int n = gguf_get_n_kv(ctx);
    for (int i = 0; i < n; i++) {
        if (std::string(gguf_get_key(ctx, i)) == key) {
            return i;
        }
    }
    return -1;
}

// quantize a model
bool dino_model_quantize(const char* fname_inp, const char* fname_out, int itype) {
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    case 6:
        type = GGML_TYPE_Q5_0;
        break;
    case 7:
        type = GGML_TYPE_Q5_1;
        break;
    case 8:
        type = GGML_TYPE_Q8_0;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return false;
    };

    struct gguf_init_params ip = {
        /* .no_alloc = */ false,
        /* .ctx      = */ nullptr,
    };

    struct gguf_context* ctx = gguf_init_from_file(fname_inp, ip);
    if (!ctx) {
        std::fprintf(stderr, "Failed to load GGUF file '%s'\n", fname_inp);
        return false;
    }

    {
        int idx = find_kv_index(ctx, "0");
        if (idx >= 0) {
            std::map<int, std::string> id2label;
            const int n_kv = gguf_get_n_kv(ctx);

            for (int i = 0; i < n_kv; i++) {
                const char* key = gguf_get_key(ctx, i);

                bool is_num = true;
                for (const char* p = key; *p; p++) {
                    if (*p < '0' || *p > '9') { is_num = false; break; }
                }
                if (!is_num) continue;
                const int label_id = std::stoi(key);
                const char* lbl = gguf_get_val_str(ctx, i);
                id2label[label_id] = lbl;
            }
        }
    }

    const std::vector<const char*> hkeys = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_classes",
        "patch_size",
        "img_size",
        "ftype",
        "num_register_tokens"
    };

    std::puts("Hyperparameters:");
    for (auto key : hkeys) {
        int i = find_kv_index(ctx, key);
        if (i < 0) {
            std::printf("  %-20s : <not found>\n", key);
            continue;
        }
        const uint32_t v = gguf_get_val_u32(ctx, i);
        std::printf("  %-20s : %u\n", key, v);
    }


    if (!gguf_write_to_file(ctx, fname_out, /*only_meta=*/false)) {
        fprintf(stderr, "Error: could not write quantized model to '%s'\n", fname_out);
        gguf_free(ctx);
        return false;
    }


    gguf_free(ctx);

    return true;

}

// usage:
// ./quantize models/ggml-model-f16.gguf models/ggml-model-f16-quant.gguf 2
//

int main(int argc, char** argv) {

    const char* fname_inp = argv[1];
    const char* fname_out = argv[2];
    const int itype = atoi(argv[3]);

    
    if (!dino_model_quantize(fname_inp, fname_out, itype)) {
        fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp);
        return 1;
    }

    
    return 0;

    
}