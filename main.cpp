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

#include "ggml-backend.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;

    image_u8 img0;
    image_f32 img1;

    dino_model model;

    int64_t t_load_us = 0;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!dino_model_load(params.model, model, params)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // load the image
    if (!load_image_from_file(params.fname_inp, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess the image to f32
    if (dino_image_preprocess(img0, img1, model.hparams)) {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    }

    // prepare for graph computation, memory allocation and results processing
    {
        // printf("%s: Initialized context = %ld bytes\n", __func__, buf_size);
        // } {


        // run prediction on img1
        const int64_t t_start_ms = ggml_time_ms();
        std::unique_ptr<dino_output> output = dino_predict(model, img1, params);

        const int64_t t_predict_ms = ggml_time_ms() - t_start_ms;

        // report timing

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: forward pass time = %8.2lld ms\n", __func__,
                t_predict_ms);
    }


    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
