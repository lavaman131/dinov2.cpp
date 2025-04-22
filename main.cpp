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
#include <iostream>
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

    // load the image
    cv::Mat img = cv::imread(params.fname_inp, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img.size[0], img.size[1]);


    // load the model
    if (!dino_model_load(img.size(), params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    if (params.classify)
        img = dino_classify_preprocess(img, img.size(), model.hparams);
    else
        img = dino_preprocess(img, img.size(), model.hparams);

    cv::Size original_size = img.size();

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img.size[0], img.size[1]);


    // prepare for graph computation, memory allocation and results processing
    {
        ggml_backend_synchronize(model.backend);
        int64_t start_time = ggml_time_ms();
        std::unique_ptr<dino_output> output = dino_predict(model, img, params);
        ggml_backend_synchronize(model.backend);
        int64_t end_time = ggml_time_ms();
        fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, end_time - start_time);

        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);


        if (!params.classify) {
            const cv::Mat &patch_tokens = output->patch_tokens.value();
            cv::PCA pca(patch_tokens, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);

            // project original features into the new 3‑D PCA space
            cv::Mat projected;
            pca.project(patch_tokens, projected);
            // projected: total_pixels×3, CV_32F

            cv::Mat projected_norm;
            cv::normalize(projected, projected_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

            cv::Mat image = projected_norm.reshape(3, img.rows / model.hparams.patch_size);

            cv::Mat resized_image;
            cv::resize(image, resized_image, original_size, 0, 0, cv::INTER_NEAREST);

            const std::string filename = params.image_out;
            if (cv::imwrite(filename, resized_image)) {
                fprintf(stderr, "%s: Saved image to: %s\n", __func__, filename.c_str());
            } else {
                fprintf(stderr, "%s: failed to save image to '%s'\n", __func__, filename.c_str());
            }
        }
    }


    return 0;
}
