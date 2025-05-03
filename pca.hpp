#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#include "ggml-cpu.h"

namespace PCA {
    // input params for PCA computations
    struct pca_params {
        int n_threads = 1;
        int n_batch = 32;
        int n_iterations = 100;
        float tolerance = 1e-6;

        int n_components = 3;

        // for debugging
        int i_layer = 0;
        int n_layers = 0;
    };

    // result from each iteration
    struct pca_result {
        struct ggml_tensor *calculated_square = nullptr;
        std::vector<struct ggml_tensor *> eigenvectors;
        std::vector<float> distances;
    };

    struct pca_model {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_t buffer;
        struct ggml_context *ctx; // context to compute graph on target device
        struct ggml_context *ctx_host; // host context to store results

        // tensors on target device
        struct ggml_tensor *dev_input;
        struct ggml_tensor *dev_square;
        struct ggml_tensor *dev_eigenvector;

        pca_model(const struct ggml_tensor *t_input) {
#ifdef GGML_USE_CUDA
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        backend = ggml_backend_cuda_init(0); // init device 0
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
#endif

#ifdef GGML_USE_METAL
            fprintf(stderr, "%s: using Metal backend\n", __func__);
            backend = ggml_backend_metal_init();
            if (!backend) {
                fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
            }
#endif

            // if there aren't GPU Backends fallback to CPU backend
            if (!backend) {
                backend = ggml_backend_cpu_init();
            }

            const int num_tensors = 4;
            struct ggml_init_params params{
                /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx = ggml_init(params);

            auto n_samples = t_input->ne[0];
            auto n_embd = t_input->ne[1];

            dev_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, n_embd);
            dev_square = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            dev_eigenvector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            ggml_set_name(dev_input, "dev_input");
            ggml_set_name(dev_square, "dev_square");
            ggml_set_name(dev_eigenvector, "dev_eigenvector");
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(dev_input, t_input->data, 0, ggml_nbytes(t_input));

            // initialize eigenvector to random normalized vector
            {
                std::vector<float> random_vec(ggml_nelements(dev_eigenvector), 0.0);
                std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
                std::uniform_real_distribution<float> distribution(0.0, 1.0);
                float sum_sqr = 0.0; // for normalizing random_vec
                for (float &i: random_vec) {
                    float f = distribution(generator);
                    sum_sqr += f * f;
                    i = f;
                }
                // normalize it
                float random_vec_norm = std::sqrt(sum_sqr);
                for (float &i: random_vec) {
                    i /= random_vec_norm;
                }
                ggml_backend_tensor_set(dev_eigenvector, random_vec.data(), 0, ggml_nbytes(dev_eigenvector));
            }
        }

        ~pca_model() {
            ggml_free(ctx);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
        }
    };

    static struct ggml_cgraph *build_graph_piter(
        const struct pca_params &params,
        const pca_model &model,
        const bool calc_square = false) {
        GGML_ASSERT(params.n_batch > 0);
        // TODO: buf_size must be able to scale with params.n_batch
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params0 = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // create a temporally context to build the graph
        struct ggml_context *ctx0 = ggml_init(params0);
        struct ggml_cgraph *gf = ggml_new_graph(ctx0);

        // turn v_diff_original into square matrix if needed
        struct ggml_tensor *tmp_square;
        if (calc_square) {
            tmp_square = ggml_mul_mat(ctx0, model.dev_input, model.dev_input);
            ggml_set_name(tmp_square, "tmp_square");
        }

        struct ggml_tensor *b_tensor;
        struct ggml_tensor *distance;
        struct ggml_tensor *old_eigen = model.dev_eigenvector;
        struct ggml_tensor *input_square = calc_square ? tmp_square : model.dev_square;

        for (int i = 0; i < params.n_batch; ++i) {
            // b_tensor = square * eigenvector^T
            b_tensor = ggml_mul_mat(ctx0, input_square, old_eigen);
            ggml_set_name(b_tensor, "b_tensor");

            // normalize
            b_tensor = ggml_div_inplace(ctx0,
                                        b_tensor,
                                        ggml_sqrt_inplace(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, b_tensor)))
            );
            ggml_format_name(b_tensor, "b_tensor_norm_%d", i);

            // calculate distance(new eigenvector - old eigenvector)
            // we don't use ggml_sub because it may not be implemented on GPU backend
            struct ggml_tensor *new_sub_old = ggml_add(ctx0, old_eigen, ggml_scale(ctx0, b_tensor, -1));
            distance = ggml_sqrt_inplace(ctx0,
                                         ggml_sum_rows(ctx0, ggml_sqr_inplace(ctx0, new_sub_old)));
            ggml_format_name(distance, "distance_%d", i);

            old_eigen = b_tensor;

            // build operations nodes
            ggml_build_forward_expand(gf, distance);
        }

        // delete the temporally context used to build the graph
        ggml_free(ctx0);
        return gf;
    }

    static ggml_status compute_piter(
        const struct pca_params &params,
        const pca_model &model,
        struct ggml_cgraph *gf,
        ggml_gallocr_t allocr,
        struct pca_result &result) {
        // allocate tensors
        ggml_gallocr_alloc_graph(allocr, gf);

        if (ggml_backend_is_cpu(model.backend)) {
            ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
        }

        const ggml_status res = ggml_backend_graph_compute(model.backend, gf);
        if (res == GGML_STATUS_SUCCESS) {
            auto extract_i = [](std::string prefix, std::string str) -> int {
                int i = -1;
                if (str.rfind(prefix, 0) == 0) {
                    sscanf(str.c_str(), (prefix + "%d").c_str(), &i);
                }
                return i;
            };
            result.calculated_square = nullptr;
            result.eigenvectors.clear();
            result.distances.clear();
            result.eigenvectors.resize(params.n_batch);
            result.distances.resize(params.n_batch);
            // get output nodes
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto node = ggml_graph_node(gf, i);
                int iter = -1;
                // find b_tensor (without copying data from device)
                if ((iter = extract_i("b_tensor_norm_", node->name)) > -1) {
                    result.eigenvectors[iter] = node;
                }
                // find distances, then copy data from device
                if ((iter = extract_i("distance_", node->name)) > -1) {
                    float d;
                    ggml_backend_tensor_get(node, &d, 0, sizeof(float));
                    result.distances[iter] = d;
                    // std::cout << node->name << " = " << d << "\n";
                }
                // find tmp_square if it exists (without copying data from device)
                if (std::string(node->name) == "tmp_square") {
                    result.calculated_square = node;
                }
            }
        }
        return res;
    }

    static cv::Mat power_iteration(
        const struct pca_params &params,
        struct ggml_tensor *input // shape [ N, D ]
    ) {
        // number of samples and embedding dimension
        const size_t N = input->ne[0];
        const size_t D = input->ne[1]; // ← fix: true vector length
        const size_t n_components = params.n_components;

        // build your PCA model (holds dev_square, dev_eigenvector, etc)
        struct pca_model model(input);

        // allocator for your power-iter graph
        ggml_gallocr_t allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend)
        );

        const int num_tensors = 3;
        struct ggml_init_params t_params{
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context *t_ctx = ggml_init(t_params);

        struct ggml_tensor *X = ggml_new_tensor_2d(t_ctx, GGML_TYPE_F32, N, D);
        struct ggml_tensor *V = ggml_new_tensor_2d(t_ctx, GGML_TYPE_F32, n_components, D);

        ggml_set_name(X, "X");
        ggml_set_name(V, "V");

        ggml_backend_buffer *t_buffer = ggml_backend_alloc_ctx_tensors(t_ctx, model.backend);

        ggml_backend_tensor_set(X, input->data, 0, ggml_nbytes(input));

        for (size_t comp = 0; comp < n_components; ++comp) {
            struct pca_params layer_params = params;
            struct pca_result result;
            struct ggml_tensor *last_eigenvector = nullptr;

            // decide how many mini-batches / iterations you want
            const int n_iters = std::max(params.n_iterations / params.n_batch, 1);

            for (int iter = 0; iter < n_iters; ++iter) {
                const bool calc_square = (iter == 0);
                // build & run one step of your power iteration
                struct ggml_cgraph *gf = build_graph_piter(layer_params, model, calc_square);
                compute_piter(layer_params, model, gf, allocr, result);

                // pick the first eigenvector whose distance < tolerance,
                // or fall back to the last one
                for (size_t k = 0; k < result.eigenvectors.size(); ++k) {
                    if (result.distances[k] < params.tolerance) {
                        last_eigenvector = result.eigenvectors[k];
                        break;
                    }
                }
                if (!last_eigenvector && !result.eigenvectors.empty()) {
                    last_eigenvector = result.eigenvectors.back();
                }

                if (calc_square) {
                    // store the gram-matrix approximation for next round
                    GGML_ASSERT(result.calculated_square != nullptr);
                    ggml_backend_tensor_copy(result.calculated_square,
                                             model.dev_square);
                } else {
                    // copy this iterate’s eigenvector onto device
                    GGML_ASSERT(last_eigenvector != nullptr);
                    ggml_backend_tensor_copy(last_eigenvector,
                                             model.dev_eigenvector);
                }
            }

            // must have converged to something
            GGML_ASSERT(last_eigenvector != nullptr);


            // assume V is shape [ n_components (rows), D (cols) ]
            size_t elem_size = ggml_element_size(V); // == sizeof(float)
            size_t row_bytes = V->ne[1] * elem_size; // D * sizeof(float)

            // pointer to the start of row `comp`
            uint8_t *dst = (uint8_t *) V->data + comp * row_bytes;

            // now copy your D‐length eigenvector into that row:
            ggml_backend_tensor_get(
                last_eigenvector,
                dst,
                /*src_offset=*/0,
                /*nbytes=*/ V->ne[1] * elem_size
            );
        }

        // --- now build the projection graph: projected = X * V ---

        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params init_params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ buf.data(),
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context *ctx = ggml_init(init_params);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);

        X = ggml_cont(ctx, ggml_transpose(ctx, X));
        V = ggml_cont(ctx, ggml_transpose(ctx, V));

        // projected: [ N, n_components ]
        struct ggml_tensor *projected = ggml_mul_mat(ctx, X, V);
        ggml_set_output(projected);
        ggml_build_forward_expand(gf, projected);

        // allocate and bind data
        ggml_gallocr_t allocr2 = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend)
        );
        ggml_gallocr_alloc_graph(allocr2, gf);

        // run it
        if (ggml_backend_graph_compute(model.backend, gf) !=
            GGML_STATUS_SUCCESS) {
            fprintf(stderr, "power_iteration: projection compute failed\n");
        }

        // pull back the result into a cv::Mat
        cv::Mat projected_mat(N, n_components, CV_32F);
        ggml_backend_tensor_get(
            projected,
            projected_mat.data,
            0,
            ggml_nbytes(projected)
        );

        // clean up
        ggml_free(ctx);
        ggml_gallocr_free(allocr2);
        ggml_gallocr_free(allocr);
        ggml_free(t_ctx);
        ggml_backend_buffer_free(t_buffer);

        return projected_mat;
    }


    static cv::Mat run_pca(
        const struct pca_params &params_in,
        const dino_hparams &hparams,
        struct ggml_tensor *input, // [n_samples, n_embd]
        const cv::Size img_size
    ) {
        // make a local copy so we can tweak i_layer / n_layers
        struct pca_params params = params_in;

        // printf("%s: computing %d components on %d-d data…\n",
        //        __func__, n_components, n_embd);

        cv::Mat projected = power_iteration(params, input);

        cv::Mat projected_norm;
        cv::normalize(projected, projected_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(projected_norm, projected_norm, cv::COLOR_GRAY2RGB);
        cv::Mat image = projected_norm.reshape(3, img_size.height / hparams.patch_size);

        cv::Mat resized_image;
        cv::resize(image, resized_image, img_size, 0, 0, cv::INTER_NEAREST);

        return resized_image;
    }
}
