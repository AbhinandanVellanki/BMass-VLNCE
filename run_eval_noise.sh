#!/bin/bash
CPT=ckpt.14.pth
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_20/run_1
CPT_PATH=$CPT_FOLDER/$CPT
CONFIG_FILE=/home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_obs.yaml

echo "🔹 Running evaluation with NOISY observations (Gaussian noise)..."
echo ""

xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals_noisy_gaussian \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.USE_NOISE True \
EVAL.NOISE.RGB_NOISE_TYPE "gaussian" \
EVAL.NOISE.DEPTH_NOISE_TYPE "gaussian" \
EVAL.NOISE.RGB_STD 0.25 \
EVAL.NOISE.DEPTH_STD 0.25