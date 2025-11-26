CPT=ckpt.14.pth
RUN=1
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_20/run_$RUN
CPT_PATH=$CPT_FOLDER/$CPT
# VAL SEEN Evaluation

CLEAN_CONFIG_FILE=/home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml

# clean
echo "🔹 Running evaluation with CLEAN observations..."
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER/ \
EVAL_CKPT_PATH_DIR  $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals/clean_images/ \
INFERENCE.SPLIT val_seen \
INFERENCE.LANGUAGES "['en-US']" \
INFERENCE.CKPT_PATH $CPT_FOLDER/$CPT \
INFERENCE.PREDICTIONS_FILE rxr_cma_en_val_seen_challenge.jsonl \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
LOG_FILE "data/logs/rxr_cma_en_20/eval_clean.log"


# noisy

Noise_CONFIG_FILE=/home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_obs.yaml

echo "🔹 Running evaluation with NOISY observations (Gaussian noise 10%)"
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $Noise_CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals/images_noisy_gaussian_0.10 \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.USE_NOISE True \
EVAL.NOISE.RGB_NOISE_TYPE "gaussian" \
EVAL.NOISE.DEPTH_NOISE_TYPE "gaussian" \
EVAL.NOISE.RGB_STD 0.10 \
EVAL.NOISE.DEPTH_STD 0.10 \
LOG_FILE "data/logs/rxr_cma_en_20/run_$RUN/eval_noisy_gaussian_0.10.log"


echo "🔹 Running evaluation with NOISY observations (Gaussian noise 25%)"
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals/images_noisy_gaussian_0.25 \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.USE_NOISE True \
EVAL.NOISE.RGB_NOISE_TYPE "gaussian" \
EVAL.NOISE.DEPTH_NOISE_TYPE "gaussian" \
EVAL.NOISE.RGB_STD 0.25 \
EVAL.NOISE.DEPTH_STD 0.25 \
LOG_FILE "data/logs/rxr_cma_en_20/run_$RUN/eval_noisy_gaussian_0.25.log"


echo "🔹 Running evaluation with NOISY observations (Gaussian noise 50%)"
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $Noise_CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals/images_noisy_gaussian_0.50 \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.USE_NOISE True \
EVAL.NOISE.RGB_NOISE_TYPE "gaussian" \
EVAL.NOISE.DEPTH_NOISE_TYPE "gaussian" \
EVAL.NOISE.RGB_STD 0.50 \
EVAL.NOISE.DEPTH_STD 0.50 \
LOG_FILE "data/logs/rxr_cma_en_20/run_$RUN/eval_noisy_gaussian_0.50.log"

echo "🔹 Running evaluation with NOISY observations (Gaussian noise 75%)"
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config $Noise_CONFIG_FILE \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER \
EVAL_CKPT_PATH_DIR $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals/images_noisy_gaussian_0.75 \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 1 \
EVAL.USE_NOISE True \
EVAL.NOISE.RGB_NOISE_TYPE "gaussian" \
EVAL.NOISE.DEPTH_NOISE_TYPE "gaussian" \
EVAL.NOISE.RGB_STD 0.75 \
EVAL.NOISE.DEPTH_STD 0.75 \
LOG_FILE "data/logs/rxr_cma_en_20/run_$RUN/eval_noisy_gaussian_0.75.log"




