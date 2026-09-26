#!/bin/bash

# Array of configuration files
CONFIGS=(
#   "../configs/fine-tune/echr/echr-qwen3-1.7b-scrubbed.yml"
#   "../configs/fine-tune/echr/echr-qwen3-1.7b-undefended.yml"
  # "../configs/fine-tune/echr/echr-gemma2-2b-scrubbed.yml"
#   "../configs/fine-tune/echr/echr-gemma2-2b-undefended.yml"
  # "../configs/fine-tune/echr/echr-llama-3-3b-scrubbed.yml"
  # "../configs/fine-tune/echr/echr-llama-3-3b-undefended.yml"
  "../configs/fine-tune/yelp/yelp-llama-3-3b-undefended.yml"
)

# Loop through each config file
for CONFIG_PATH in "${CONFIGS[@]}"; do
    echo "Starting run with config: $CONFIG_PATH"
    python fine_tune.py --config_path "$CONFIG_PATH"

    if [ $? -ne 0 ]; then
        echo "Run with $CONFIG_PATH failed, stopping script."
        exit 1
    fi
    echo "Completed run with $CONFIG_PATH"
done

echo "All runs finished."

# chmod +x run_multiple_configs.sh
# ./run_multiple_configs.sh