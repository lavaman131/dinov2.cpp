#!/usr/bin/env bash

# Setup platform-independent date and time commands
setup_commands() {
    OS=$(uname -s)
    if [[ "$OS" == "Darwin" ]]; then
        if command -v gdate >/dev/null 2>&1 && command -v gtime >/dev/null 2>&1; then
            echo "Using GNU tools on macOS"
        else
            echo "Error: GNU date and time not found. Install with:"
            echo "  brew install coreutils gnu-time"
            exit 1
        fi
        get_time() { gdate +%s%N; }
        measure_cmd() {
            # run with gtime, capture max RSS in mem.txt
            gtime -f "%M" -o mem.txt $1 > /dev/null 2>&1
        }
    else
        echo "Using native GNU tools on Linux"
        get_time() { date +%s%N; }
        measure_cmd() {
            # run with /usr/bin/time, capture max RSS in mem.txt
            /usr/bin/time -f "%M" -o mem.txt $1 > /dev/null 2>&1
        }
    fi
}

# models & quant settings
declare -a models=("small" "base" "large" "giant")
declare -a quant_names=("q4_0" "q4_1" "q5_0" "q5_1" "q8_0")
declare -a quant_ids=(2 3 6 7 8)
declare -A speed_results
declare -A memory_results

num_threads=12
quantize_flag=0
N=10

if [ "$#" -ge 1 ]; then
    num_threads=$1; echo "num_threads=$num_threads"
fi
if [ "$#" -ge 2 ]; then
    quantize_flag=$2; echo "quantize_flag=$quantize_flag"
fi

setup_commands

for model in "${models[@]}"; do
    model_name="facebook/dinov2-${model}-imagenet1k-1-layer"
    echo "Converting model: $model_name"
    python ./scripts/dinov2-to-gguf.py --model_name "$model_name" > /dev/null 2>&1

    cd build/ || exit

    if [ "$quantize_flag" -eq 1 ]; then
        for i in "${!quant_ids[@]}"; do
            q="${quant_names[$i]}"
            idx="${quant_ids[$i]}"
            echo "Quantizing to ${q} (index ${idx})"
            ./bin/quantize ../ggml-model.gguf ../ggml-model-quant.gguf ${idx} > /dev/null 2>&1

            sum_ms=0
            sum_mem=0

            for ((run=1; run<=N; run++)); do
                # 1) Memory
                measure_cmd "./bin/inference -c -m ../ggml-model-quant.gguf -i ../assets/tench.jpg -t $num_threads"
                mem_run=$(<mem.txt)
                sum_mem=$((sum_mem + mem_run))

                # 2) Graph time (ms)
                time_ms=$(
                  ./bin/inference -c -m ../ggml-model-quant.gguf -i ../assets/tench.jpg -t $num_threads \
                    2>&1 \
                    | sed -En 's/.*main: graph computation took ([0-9]+) ms.*/\1/p'
                )
                sum_ms=$((sum_ms + time_ms))
            done

            avg_mem=$((sum_mem / N / 1024))    # in MB
            avg_time=$((sum_ms / N))           # in ms

            speed_results["$model,$q"]=$avg_time
            memory_results["$model,$q"]=$avg_mem

            rm mem.txt
        done

    else
        echo "No quantization for model $model"

        sum_ms=0
        sum_mem=0

        for ((run=1; run<=N; run++)); do
            measure_cmd "./bin/inference -c -m ../ggml-model.gguf -i ../assets/tench.jpg -t $num_threads"
            mem_run=$(<mem.txt)
            sum_mem=$((sum_mem + mem_run))

            time_ms=$(
              ./bin/inference -c -m ../ggml-model.gguf -i ../assets/tench.jpg -t $num_threads \
                2>&1 \
                | sed -En 's/.*main: graph computation took ([0-9]+) ms.*/\1/p'
            )
            sum_ms=$((sum_ms + time_ms))
        done

        avg_mem=$((sum_mem / N / 1024))
        avg_time=$((sum_ms / N))

        speed_results["$model"]=$avg_time
        memory_results["$model"]=$avg_mem

        rm mem.txt
    fi

    cd ..
done

# Print table
if [ "$quantize_flag" -eq 1 ]; then
    echo "| Model | Quant | Speed (ms) | Mem (MB) |"
    echo "|:-----:|:------:|:----------:|:--------:|"
    for model in "${models[@]}"; do
      for j in "${!quant_ids[@]}"; do
        q="${quant_names[$j]}"
        key="$model,$q"
        if [ -n "${speed_results[$key]}" ]; then
          echo "| $model | $q | ${speed_results[$key]} | ${memory_results[$key]} |"
        fi
      done
    done
else
    echo "| Model | Speed (ms) | Mem (MB) |"
    echo "|:-----:|:----------:|:--------:|"
    for model in "${models[@]}"; do
      echo "| $model | ${speed_results[$model]} | ${memory_results[$model]} |"
    done
fi
