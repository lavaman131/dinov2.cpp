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


    img = dino_image_preprocess(img, img.size(), model.hparams);

    cv::Size original_size = img.size();

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

            cv::imshow("image", resized_image);
            cv::waitKey(0);
            cv::destroyAllWindows();


            // normalize each component to [0,255] and reshape back to H×W
            // std::vector<cv::Mat> pcs(3);
            // for (int i = 0; i < 3; ++i) {
            //     cv::Mat comp = projected.col(i); // the i-th PC vector (total_pixels×1)
            //     cv::Mat comp_norm;
            //     cv::normalize(comp, comp_norm, 0, 255, cv::NORM_MINMAX); // may be unnecessary
            //     pcs[i] = comp_norm.reshape(1, rows); // now rows×cols single‑channel
            //     pcs[i].convertTo(pcs[i], CV_8U);
            // }
            //
            // // merge into a 3‑channel BGR image
            // // we want PC0→R, PC1→G, PC2→B, but OpenCV is BGR, so:
            // std::vector<cv::Mat> bgr = {pcs[2], pcs[1], pcs[0]};
            // cv::Mat pca_image;
            // cv::merge(bgr, pca_image);

            // save frame to disk
            // cv::imwrite("pca_result.png", pca_image);
            // fprintf(stderr, "PCA result saved to 'pca_result.png' (%d×%d)\n",
            //         pca_image.cols, pca_image.rows);
        }
    }


    return 0;
}
