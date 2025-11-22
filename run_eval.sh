CPT=ckpt.3.pth
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_20/
CPT_PATH=$CPT_FOLDER/$CPT

# python run.py \
#   --run-type eval \
#   --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_eval.yaml \
# TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
# CHECKPOINT_FOLDER $CPT_FOLDER/ \
# EVAL_CKPT_PATH_DIR  $CPT_PATH \
# RESULTS_DIR $CPT_FOLDER/evals \
# INFERENCE.SPLIT val_unseen \
# INFERENCE.LANGUAGES "['en-US']" \
# INFERENCE.CKPT_PATH $CPT_FOLDER/$CPT \
# INFERENCE.PREDICTIONS_FILE rxr_cma_en_val_unseen_challenge.jsonl \
# EVAL.SPLIT val_unseen \
# EVAL.LANGUAGES "['en-US']" \
# NUM_ENVIRONMENTS 3


# python run.py \
#   --run-type eval \
#   --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_eval.yaml \
# TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
# CHECKPOINT_FOLDER $CPT_FOLDER/ \
# EVAL_CKPT_PATH_DIR  $CPT_PATH \
# RESULTS_DIR $CPT_FOLDER/evals \
# INFERENCE.SPLIT val_seen \
# INFERENCE.LANGUAGES "['en-US']" \
# INFERENCE.CKPT_PATH $CPT_FOLDER/$CPT \
# INFERENCE.PREDICTIONS_FILE rxr_cma_en_val_seen_challenge.jsonl \
# EVAL.SPLIT val_seen \
# EVAL.LANGUAGES "['en-US']" \
# NUM_ENVIRONMENTS 10

python run.py \
  --run-type eval \
  --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_eval.yaml \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER/ \
EVAL_CKPT_PATH_DIR  $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals_hin \
INFERENCE.SPLIT val_unseen \
INFERENCE.LANGUAGES "['hi-IN']" \
INFERENCE.CKPT_PATH $CPT_FOLDER/$CPT \
INFERENCE.PREDICTIONS_FILE rxr_cma_en_val_unseen_challenge.jsonl \
EVAL.SPLIT val_unseen \
EVAL.LANGUAGES "['hi-IN']" \
NUM_ENVIRONMENTS 3


python run.py \
  --run-type eval \
  --exp-config /home/abhi/Documents/VLN-CE-py38/vlnce_baselines/config/rxr_baselines/rxr_cma_en_eval.yaml \
TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en \
CHECKPOINT_FOLDER $CPT_FOLDER/ \
EVAL_CKPT_PATH_DIR  $CPT_PATH \
RESULTS_DIR $CPT_FOLDER/evals_hin \
INFERENCE.SPLIT val_seen \
INFERENCE.LANGUAGES "['hi-IN']" \
INFERENCE.CKPT_PATH $CPT_FOLDER/$CPT \
INFERENCE.PREDICTIONS_FILE rxr_cma_en_val_seen_challenge.jsonl \
EVAL.SPLIT val_seen \
EVAL.LANGUAGES "['hi-IN']" \
NUM_ENVIRONMENTS 10

