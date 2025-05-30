﻿#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "realtime.h"
#include "dinov2.h"
#include "ggml.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "ggml/examples/stb_image.h" // stb image load
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <cinttypes>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "ggml-backend.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

int main(int argc, char **argv) {
    dino_params params;
    dino_model model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    cv::VideoCapture cap(params.camera_id);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    cv::Mat frame;

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    auto size = cv::Size(FRAME_WIDTH, FRAME_HEIGHT);

    // load the model
    if (!dino_model_load(size, params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    const auto new_size = cv::Size((FRAME_WIDTH / model.hparams.patch_size + 1) * model.hparams.patch_size,
                                   (FRAME_HEIGHT / model.hparams.patch_size + 1) * model.hparams.patch_size);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        cv::resize(frame, frame, size, 0, 0, cv::INTER_NEAREST);

        // load the image
        cv::Mat input = dino_preprocess(frame, size, model.hparams);

        // output from model
        ggml_backend_synchronize(model.backend);
        int64_t start_time = ggml_time_ms();
        std::unique_ptr<dino_output> output = dino_predict(model, input, params, allocr);
        ggml_backend_synchronize(model.backend);
        int64_t end_time = ggml_time_ms();
        fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, end_time - start_time);

        // pca conversion
        const cv::Mat &patch_tokens = output->patch_tokens.value();
        cv::PCA pca(patch_tokens, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);

        // project original features into the new 3‑D PCA space
        cv::Mat projected;
        pca.project(patch_tokens, projected);
        // projected: total_pixels×3, CV_32F

        cv::Mat projected_norm;
        cv::normalize(projected, projected_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::Mat pca_image = projected_norm.reshape(3, new_size.height / model.hparams.patch_size);

        cv::resize(pca_image, pca_image, frame.size(), 0, 0, cv::INTER_NEAREST);

        cv::Mat combined_frame;
        std::vector<cv::Mat> imgs = {frame, pca_image};
        cv::hconcat(imgs, combined_frame);

        cv::imshow("Output", combined_frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    ggml_free(model.ctx);
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
