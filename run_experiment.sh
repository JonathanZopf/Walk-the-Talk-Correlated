#!/bin/bash
set -e  # exit on error

# Default values
N_EXAMPLES=""
DATASET=""
MODEL=""
MAX_GROUPS=""
ONLY_CONCEPT_REMOVALS=""
COUNTERFACTUAL_GEN_BASE_PROMPT_NAME=""

# Usage
usage() {
    echo "Usage: $0 -n <num_examples> -d <dataset> -m <model> -g <max_groups> [-r] [-p <prompt_name>]"
    echo "  -r : enable only_concept_removals (flag, no argument)"
    echo "  -p : counterfactual_gen_base_prompt_name (optional)"
    echo "Example: $0 -n 30 -d bbq -m gpt-oss:120b -g 4 -r -p counterfactual_gen_replacements_prompt.txt"
    exit 1
}

# Parse command line arguments
while getopts "n:d:m:g:rp:" opt; do
    case $opt in
        n) N_EXAMPLES="$OPTARG" ;;
        d) DATASET="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        g) MAX_GROUPS="$OPTARG" ;;
        r) ONLY_CONCEPT_REMOVALS="true" ;;   # flag set
        p) COUNTERFACTUAL_GEN_BASE_PROMPT_NAME="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check mandatory arguments
if [ -z "$N_EXAMPLES" ] || [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ -z "$MAX_GROUPS" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Construct base paths
BASE_DIR="output"
INTERVENTION_DIR="${BASE_DIR}/intervention_generation/${DATASET}/${MODEL}_${MAX_GROUPS}"
MODEL_RESP_DIR="${BASE_DIR}/model_responses/${DATASET}/${MODEL}_${MAX_GROUPS}"
IMPLIED_DIR="${BASE_DIR}/implied_concepts/${DATASET}/${MODEL}_${MAX_GROUPS}"
FAITHFULNESS_DIR="${BASE_DIR}/faithfulness_results/${DATASET}/${MODEL}_${MAX_GROUPS}"

echo "=========================================="
echo "Running experiment with:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Number of examples: $N_EXAMPLES"
echo "  Max correlated concepts per group: $MAX_GROUPS"
echo "  Only concept removals: ${ONLY_CONCEPT_REMOVALS:-false}"
echo "  Counterfactual prompt name: ${COUNTERFACTUAL_GEN_BASE_PROMPT_NAME:-<default>}"
echo "=========================================="

# 1. Generate interventions
echo "Step 1: Generating interventions..."
CMD="python src/run_generate_interventions.py \
    --dataset=\"$DATASET\" \
    --dataset_path=\"data/${DATASET}\" \
    --intervention_model=\"$MODEL\" \
    --n_examples=\"$N_EXAMPLES\" \
    --n_workers=5 \
    --output_dir=\"$INTERVENTION_DIR\" \
    --max_interventions_in_correlation_groups=\"$MAX_GROUPS\""

# Conditionally add flags/arguments
if [ -n "$ONLY_CONCEPT_REMOVALS" ]; then
    CMD="$CMD --only_concept_removals"
fi
if [ -n "$COUNTERFACTUAL_GEN_BASE_PROMPT_NAME" ]; then
    CMD="$CMD --counterfactual_gen_base_prompt_name=\"$COUNTERFACTUAL_GEN_BASE_PROMPT_NAME\""
fi

eval $CMD

# 2. Collect model responses
echo "Step 2: Collecting model responses..."
python src/run_collect_model_responses.py \
    --dataset="$DATASET" \
    --dataset_path="data/${DATASET}" \
    --language_model="$MODEL" \
    --cot \
    --few_shot \
    --few_shot_prompt_name="few_shot_cot_prompt" \
    --n_examples="$N_EXAMPLES" \
    --intervention_data_path="$INTERVENTION_DIR" \
    --output_dir="$MODEL_RESP_DIR" \
    --n_completions=5

# 3. Determine implied concepts
echo "Step 3: Determining implied concepts..."
python src/run_determine_implied_concepts.py \
    --implied_concepts_model="$MODEL" \
    --n_examples="$N_EXAMPLES" \
    --intervention_data_path="$INTERVENTION_DIR" \
    --model_response_data_path="$MODEL_RESP_DIR" \
    --output_dir="$IMPLIED_DIR" \
    --dataset="$DATASET" \
    --dataset_path="data/${DATASET}"

# 4. Measure faithfulness
echo "Step 4: Measuring faithfulness..."
python src/run_measure_faithfulness.py \
    --n_examples="$N_EXAMPLES" \
    --save_results \
    --model_name="$MODEL" \
    --print_results \
    --dataset="$DATASET" \
    --dataset_path="data/${DATASET}" \
    --intervention_dir="$INTERVENTION_DIR" \
    --model_response_dir="$MODEL_RESP_DIR" \
    --implied_concepts_dir="$IMPLIED_DIR" \
    --output_dir="$FAITHFULNESS_DIR"

echo "=========================================="
echo "Experiment completed successfully."
echo "Results saved under: $FAITHFULNESS_DIR"
echo "=========================================="