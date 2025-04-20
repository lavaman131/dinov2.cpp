#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ggml-alloc.h"
#include "ggml/examples/stb_image.h" // stb image load

#include "realtime.h"

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

#include <iostream>
#include <windows.h>
#include <conio.h> 

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

int main(int argc, char** argv) {

    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    cv::Mat frame;

    // for loading the model with interpolation
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);


    dino_params params;
    dino_model model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    if (!dino_model_load(params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    while (true) {

        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        cv::imshow("Camera Feed", frame);


        // load the image
        frame = dino_image_preprocess(frame, model.hparams);

        // output from model
        std::unique_ptr<dino_output> output = dino_predict(model, frame, params);


        // pca conversion
        const cv::Mat& patch_tokens = output->patch_tokens.value();
        cv::PCA pca(patch_tokens, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);

        // project original features into the new 3‑D PCA space
        cv::Mat projected;
        pca.project(patch_tokens, projected);
        // projected: total_pixels×3, CV_32F


        cv::Mat projected_norm;
        cv::normalize(projected, projected_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        int size = model.hparams.n_img_embd();
        cv::Mat image = projected_norm.reshape(3, size);

        cv::Size new_size = cv::Size(width, height);

        cv::Mat resized_image;
        cv::resize(image, resized_image, new_size, 0, 0, cv::INTER_NEAREST);


        cv::imshow("PCA Output", resized_image);


        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    cap.release();
    cv::destroyAllWindows();
    return 0;


    return 0;

}