#!/bin/bash

# Array of config files: ../configs/evaluate/pii-inference1.yml ... pii-inference10.yml
CONFIGS=()
for i in {1..13}; do
    # CONFIGS+=("../configs/evaluate/extraction-gated-qwen-1.7b/pii-extraction${i}.yml")
    # CONFIGS+=("../configs/evaluate/extraction-gated-3b/pii-extraction${i}.yml")
    # CONFIGS+=("../configs/evaluate/extraction-gated-1b/pii-extraction${i}.yml")
    # CONFIGS+=("../configs/evaluate/extraction-gated-gemma/pii-extraction${i}.yml")
    # CONFIGS+=("../configs/evaluate/extraction-non-fl/pii-extraction${i}.yml")

    CONFIGS+=("../configs/evaluate/extraction-gated-gemma/pii-extraction${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-gemma/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-qwen/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-1b/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-1b-4lora/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/perplxity_fl/perplxity${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-1b-12lora/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-1b-16lora/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-3b/pii-inference${i}.yml")
    # CONFIGS+=("../configs/evaluate/inference-gated-1b-4lora/pii-inference${i}.yml")
done

# TAG="llama3-3b-yelp-lora-8"
# TAG="llama3-3b-echer-lora-8"

# TAG="qwen-1.7b-yelp-lora-8"
# TAG="llama3-1b-echer-16lora-copy2"
# TAG="llama3-3b-echer"
# TAG="llama3-1b-echer-20c"
# TAG="llama3-1b-echer-fedyogi"
# TAG="llama3-1b-echer-30c"
# TAG="llama3-3b-yelp"
# TAG="llama3-1b-yelp"
# TAG="llama3-1b-echer"
# TAG="gemma-echer"
# TAG="llama3-3b-echer-lora-8-different-pii-classes"
# TAG="llama3-3b-echer-lora-8-different-c"
TAG="gemma-yelp"
# TAG="llama-1b-echer-lora-16-fl-fedyogi-20r-merged-avg"
# TAG="gemma-yelp"
# TAG="qwen-yelp"
# TAG="qwen-echr"
# TAG="extraction-non-fl"
# TAG="llama3-1b-echer-14lora-3"
# TAG="llama3-1b-echer-4lora"
# TAG="llama3-1b-echer-12lora"
# TAG="llama3-1b-echer-16lora"
# TAG="llama3-1b-echer-16lora-perplixty"
LOG_FILE="attack_run_{$TAG}_log.txt"

echo "Starting evaluations..." > "$LOG_FILE"

# Loop through each config file
for i in {9..9}; do
    CONFIG_PATH=${CONFIGS[$((i-1))]}
    # OUTPUT_FILE="attack_inference_${TAG}_${i}.txt"
    OUTPUT_FILE="attack_extraction__non_fl_${TAG}_${i}.txt"
    # OUTPUT_FILE="perplixt_non_fl_${TAG}_${i}.txt"

    echo "Starting run $i with config: $CONFIG_PATH"

    # Clear Hugging Face cache before each run
    echo "Clearing Hugging Face cache..." | tee -a "$LOG_FILE"
    rm -rf ~/.cache/huggingface/hub

    # python evaluate.py --config_path "$CONFIG_PATH" > "$OUTPUT_FILE"
    python evaluate.py --config_path "$CONFIG_PATH" > "$OUTPUT_FILE" 2>&1
    # python evaluate_perplixty.py --config_path "$CONFIG_PATH" > "$OUTPUT_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "Run $i with $CONFIG_PATH failed. Check $OUTPUT_FILE for details." | tee -a "$LOG_FILE"
        continue  # Skip to next config instead of exiting
    fi

    echo "Completed run $i" | tee -a "$LOG_FILE"
done

echo "All runs finished (some may have failed)." | tee -a "$LOG_FILE"

# chmod +x run_multiple_evaluate_configs.sh
# ./run_multiple_evaluate_configs.sh