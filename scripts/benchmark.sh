#!/usr/bin/env bash

# Setup platform-independent date and time commands
setup_commands() {
    # Detect operating system
    OS=$(uname -s)
    
    if [[ "$OS" == "Darwin" ]]; then
        # macOS requires GNU tools installed via Homebrew
        if command -v gdate >/dev/null 2>&1 && command -v gtime >/dev/null 2>&1; then
            echo "Using GNU tools on macOS"
        else
            echo "Error: GNU date and time not found. Install with:"
            echo "  brew install coreutils gnu-time"
            exit 1
        fi

        # Define wrapper functions for macOS
        get_time() {
            gdate +%s%N
        }

        measure_cmd() {
            local full_cmd="$1"
            gtime -f "%M" -o mem.txt \
                $full_cmd > /dev/null 2>&1
        }

        get_mem() {
            cat mem.txt
        }
    else
        echo "Using native GNU tools on Linux"

        # Define wrapper functions for Linux
        get_time() {
            date +%s%N
        }

        measure_cmd() {
            local full_cmd="$1"
            /usr/bin/time -f "%M" -o mem.txt \
                $full_cmd > /dev/null 2>&1
        }

        get_mem() {
            cat mem.txt
        }
    fi
}

# arrays
declare -a models=("small" "base" "large" "giant")
declare -a quant_names=("q4_0" "q4_1" "q5_0" "q5_1" "q8_0")
declare -a quant_ids=(2 3 6 7 8)
declare -A speed_results
declare -A memory_results

# defaults
num_threads=4      # (now unused)  
quantize_flag=0    # 0 for no quantization, 1 for quantization
N=10               # number of times to run each model

if [ "$#" -ge 1 ]; then
    echo "num_threads=$1"
    num_threads=$1
fi

if [ "$#" -ge 2 ]; then
    echo "quantize_flag=$2"
    quantize_flag=$2
fi

setup_commands

for model in "${models[@]}"; do
    model_name="facebook/dinov2-${model}-imagenet1k-1-layer"
    echo "Converting model: $model_name"
    python ./scripts/dinov2-to-gguf.py --model_name "$model_name" --ftype 1 > /dev/null 2>&1

    cd build/ || exit

    if [ "$quantize_flag" -eq 1 ]; then
        for i in "${!quant_ids[@]}"; do
            q="${quant_names[$i]}"
            q_index="${quant_ids[$i]}"
            echo "Quantizing ... to ${q} (index ${q_index})"
            ./bin/quantize ../ggml-model-f16.gguf ../ggml-model-f16-quant.gguf ${q_index} > /dev/null 2>&1

            sum=0
            mem_usage=0

            for ((run=1; run<=N; run++)); do
                start=$(get_time)
                measure_cmd "./bin/dinov2 -c -m ../ggml-model-f16-quant.gguf -i ../assets/tench.jpg"
                end=$(get_time)
                diff=$((end - start))
                sum=$((sum + diff))
                mem_usage=$((mem_usage + $(cat mem.txt)))
            done

            avg_mem_usage=$((mem_usage / N / 1024))
            avg_speed=$((sum / N / 1000000))

            speed_results["$model,$q"]=$avg_speed
            memory_results["$model,$q"]=$avg_mem_usage

            rm mem.txt
        done
    else
        echo "No quantization for model $model"

        sum=0
        mem_usage=0

        for ((run=1; run<=N; run++)); do
            start=$(get_time)
            measure_cmd "./bin/dinov2 -c -m ../ggml-model-f16.gguf -i ../assets/tench.jpg"
            end=$(get_time)
            diff=$((end - start))
            sum=$((sum + diff))
            mem_usage=$((mem_usage + $(cat mem.txt)))
        done

        avg_mem_usage=$((mem_usage / N / 1024))
        avg_speed=$((sum / N / 1000000))

        speed_results["$model"]=$avg_speed
        memory_results["$model"]=$avg_mem_usage

        rm mem.txt
    fi

    cd ..
done

# Print results table
if [ "$quantize_flag" -eq 1 ]; then
    echo "| Model  | Quantization | Speed (ms) | Mem (MB) |"
    echo "| :----: | :----------: | :--------: | :------: |"
    for model in "${models[@]}"; do
        for i in "${!quant_ids[@]}"; do
            q="${quant_names[$i]}"
            key="$model,$q"
            if [ -n "${speed_results[$key]}" ]; then
                echo "| $model | $q | ${speed_results[$key]} | ${memory_results[$key]} |"
            fi
        done
    done
else
    echo "| Model  | Speed (ms) | Mem (MB) |"
    echo "| :----: | :--------: | :------: |"
    for model in "${models[@]}"; do
        echo "| $model | ${speed_results[$model]} | ${memory_results[$model]} |"
    done
fi
