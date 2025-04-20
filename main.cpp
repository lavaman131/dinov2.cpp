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

        // add if statement for if the -c flag was not called
        // also add timing for PCA section below
        if (output->patch_tokens.has_value()) {
            // extract the planar feature vector (CV_32F) from the model output
            cv::Mat feat_planar = *output->patch_tokens; // size: 1×(3*H*W) or (3*H*W)×1

            // compute total pixels from original image dimensions
            int rows = img.rows;
            int cols = img.cols;
            int total_pixels = rows * cols;

            // reshape to a 2D matrix (total_pixels × 3), 
            // so each row is one pixel, columns are [R, G, B]
            cv::Mat feat_reshaped = feat_planar.reshape(1, total_pixels); // may need to do this after pca instead of before
            // now feat_reshaped is CV_32F, size = total_pixels×3

            // PCA with top 3 principal components
            cv::PCA pca(feat_reshaped, cv::Mat(), cv::PCA::DATA_AS_ROW, /*maxComponents=*/3);

            // project original features into the new 3‑D PCA space
            cv::Mat projected;
            pca.project(feat_reshaped, projected);
            // projected: total_pixels×3, CV_32F

            // normalize each component to [0,255] and reshape back to H×W
            std::vector<cv::Mat> pcs(3);
            for (int i = 0; i < 3; ++i) {
                cv::Mat comp = projected.col(i);     // the i-th PC vector (total_pixels×1)
                cv::Mat comp_norm;
				cv::normalize(comp, comp_norm, 0, 255, cv::NORM_MINMAX); // may be unnecessary
                pcs[i] = comp_norm.reshape(1, rows); // now rows×cols single‑channel
                pcs[i].convertTo(pcs[i], CV_8U);
            }

            // merge into a 3‑channel BGR image
            // we want PC0→R, PC1→G, PC2→B, but OpenCV is BGR, so:
            std::vector<cv::Mat> bgr = { pcs[2], pcs[1], pcs[0] };
            cv::Mat pca_image;
            cv::merge(bgr, pca_image);

            // save frame to disk
            cv::imwrite("pca_result.png", pca_image);
            fprintf(stderr, "PCA result saved to 'pca_result.png' (%d×%d)\n",
                pca_image.cols, pca_image.rows);

        }
        else {
            fprintf(stderr, "Warning: no feature map available for PCA\n");
        }
    }

    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
