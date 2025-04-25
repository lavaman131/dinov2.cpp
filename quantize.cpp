// add simple qunatization strategies
// adapted from : ggml/gpt-2

#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "dinov2.h"


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

#include <inttypes.h>


int main(int argc, char **argv) {
    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const auto itype = std::atoi(argv[3]);

    if (!dino_model_quantize(fname_inp, fname_out, itype)) {
        fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
        return 1;
    }

    return 0;
}
