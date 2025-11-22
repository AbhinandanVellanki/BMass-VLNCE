MODEL_NAME=rxr_cma_en_20_text_only
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints
RUN=run_1

python run.py \
  --run-type train \
  --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
TENSORBOARD_DIR data/tensorboard_dirs/$MODEL_NAME/$RUN \
CHECKPOINT_FOLDER $CPT_FOLDER/$MODEL_NAME/$RUN \
EVAL_CKPT_PATH_DIR  $CPT_PATH/evals/$MODEL_NAME/$RUN \
RESULTS_DIR $CPT_FOLDER/evals \
INFERENCE.SPLIT train \
INFERENCE.LANGUAGES "['en-US']" \
INFERENCE.CKPT_PATH $CPT_FOLDER/$MODEL_NAME/$RUN/ckpt.0.pth \
INFERENCE.PREDICTIONS_FILE rxr_cma_en_20_text_only.jsonl \
EVAL.SPLIT train \
EVAL.LANGUAGES "['en-US']" \
NUM_ENVIRONMENTS 3 \
MODEL.ablate_depth True \
MODEL.ablate_rgb True \
TASK_CONFIG.DATASET.SPLIT train \
TASK_CONFIG.DATASET.LANGUAGES "['en-US']"
