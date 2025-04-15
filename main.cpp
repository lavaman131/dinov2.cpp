#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include "ggml-alloc.h"
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
#include <thread>
#include <cinttypes>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    dino_params params;

    image_u8 img0;
    image_f32 img1;

    dino_model model;
    dino_state state;

    int64_t t_load_us = 0;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(nullptr);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);
    fprintf(stderr, "%s: n_threads = %d / %d\n", __func__, params.n_threads,
            (int32_t) std::thread::hardware_concurrency());

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!dino_model_load(params.model, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // load the image
    // if (!load_image_from_file(params.fname_inp, img0)) {
    //     fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
    //     return 1;
    // }
    // fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess the image to f32
    // if (dino_image_preprocess(img0, img1, model.hparams)) {
    //     fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    // }

    // prepare for graph computation, memory allocation and results processing
    // {
    //     static size_t buf_size = 3u * 1024 * 1024;
    //
    //     struct ggml_init_params ggml_params = {
    //         /*.mem_size   =*/buf_size,
    //         /*.mem_buffer =*/nullptr,
    //         /*.no_alloc   =*/false,
    //     };
    //
    //     state.ctx = ggml_init(ggml_params);
    //     state.prediction = ggml_new_tensor_4d(state.ctx, GGML_TYPE_F32, model.hparams.num_classes, 1, 1, 1);
    //
    //     // printf("%s: Initialized context = %ld bytes\n", __func__, buf_size);
    // // } {
    //     std::vector<std::pair<float, int> > predictions;
    //     // run prediction on img1
    //     dino_predict(model, state, img1, params, predictions);
    // }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:    model load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        fprintf(stderr, "%s:    processing time = %8.2f ms\n", __func__,
                (t_main_end_us - t_main_start_us - t_load_us) / 1000.0f);
        fprintf(stderr, "%s:    total time      = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
