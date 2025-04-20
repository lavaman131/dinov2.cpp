#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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
#include <opencv2/imgproc.hpp>

#include "ggml-backend.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;
    dino_model model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the model

    if (!dino_model_load(params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    // load the image
    cv::Mat img = cv::imread(params.fname_inp, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img.size[0], img.size[1]);

    img = dino_image_preprocess(img, model.hparams);

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img.size[0], img.size[1]);

    // prepare for graph computation, memory allocation and results processing
    {
        // run prediction on img
        const int64_t t_start_ms = ggml_time_ms();
        std::unique_ptr<dino_output> output = dino_predict(model, img, params);
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
