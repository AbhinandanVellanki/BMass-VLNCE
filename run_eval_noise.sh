#!/bin/bash
CPT=ckpt.14.pth
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_20/run_1
CPT_PATH=$CPT_FOLDER/$CPT
CONFIG_FILE=/home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_obs.yaml

# echo "🔹 Running evaluation with NOISY observations (10 patches)..."
# echo ""

# xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
#   --run-type eval \
#   --exp-config $CONFIG_FILE \
# TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
# CHECKPOINT_FOLDER $CPT_FOLDER \
# EVAL_CKPT_PATH_DIR $CPT_PATH \
# RESULTS_DIR $CPT_FOLDER/evals_noisy_patch_10 \
# EVAL.SPLIT val_seen \
# EVAL.LANGUAGES "['en-US']" \
# NUM_ENVIRONMENTS 1 \
# EVAL.RGB_NOISE_PARAMS.patch.num_patches 10 \
# EVAL.DEPTH_NOISE_PARAMS.patch.num_patches 10 \
# LOG_FILE data/logs/rxr_cma_en_20/eval_noisy_patch_10.log

# echo ""
# echo "🔹 Running evaluation with NOISY observations (25 patches)..."
# echo ""

# xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
#   --run-type eval \
#   --exp-config $CONFIG_FILE \
# TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
# CHECKPOINT_FOLDER $CPT_FOLDER \
# EVAL_CKPT_PATH_DIR $CPT_PATH \
# RESULTS_DIR $CPT_FOLDER/evals_noisy_patch_25 \
# EVAL.SPLIT val_seen \
# EVAL.LANGUAGES "['en-US']" \
# NUM_ENVIRONMENTS 1 \
# EVAL.RGB_NOISE_PARAMS.patch.num_patches 25 \
# EVAL.DEPTH_NOISE_PARAMS.patch.num_patches 25 \
# LOG_FILE data/logs/rxr_cma_en_20/eval_noisy_patch_25.log

echo ""
echo "🔹 Running evaluation with NOISY observations (50 patches)..."
echo ""

xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals_noisy_patch_50 \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.RGB_NOISE_PARAMS.patch.num_patches 50 \
EVAL.DEPTH_NOISE_PARAMS.patch.num_patches 50 \
LOG_FILE data/logs/rxr_cma_en_20/eval_noisy_patch_50.log

echo ""
echo "✅ Both evaluations completed!"
echo "📄 Logs saved to:"
echo "   - data/logs/rxr_cma_en_20/eval_noisy_patch_10.log"
echo "   - data/logs/rxr_cma_en_20/eval_noisy_patch_25.log"


