# FULL NOISE EVALUATION (text features with 50% noise added)

# # 1. Trained model
# CPT=ckpt.5.pth
# RUN=1
# CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_dirty_vision_text/run_$RUN
# CPT_PATH=$CPT_FOLDER/$CPT

# export IS_EVAL=1
# xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
#   --run-type eval \
#   --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
#   TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.features_path data/datasets/RxR_VLNCE_v0/rxr_{split}_combined_noisy_eval/{id:06}_{lang}_text_features.npz \
#   TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.noisy_features_path null \
#   TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en_dirty_vision_text/eval \
#   CHECKPOINT_FOLDER $CPT_FOLDER/ \
#   EVAL_CKPT_PATH_DIR $CPT_PATH \
#   RESULTS_DIR $CPT_FOLDER/evals_full_noise \
#   EVAL.SPLIT val_seen \
#   LOG_FILE data/logs/rxr_cma_en_dirty_vision_text/run_${RUN}/${CPT}_eval_full_noise.log

# 2. Baseline model (without noisy training)
CPT=ckpt.0.pth
RUN=1
CPT_FOLDER=/home/abhi/Documents/VLN-CE-py38/data/checkpoints/rxr_cma_en_20/run_$RUN
CPT_PATH=$CPT_FOLDER/$CPT

export IS_EVAL=1
xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type eval \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.features_path data/datasets/RxR_VLNCE_v0/rxr_{split}_combined_noisy_eval/{id:06}_{lang}_text_features.npz \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.noisy_features_path null \
  TENSORBOARD_DIR data/tensorboard_dirs/rxr_cma_en_20/eval \
  CHECKPOINT_FOLDER $CPT_FOLDER/ \
  EVAL_CKPT_PATH_DIR $CPT_PATH \
  RESULTS_DIR $CPT_FOLDER/evals_full_noise \
  EVAL.SPLIT val_seen \
  LOG_FILE data/logs/rxr_cma_en_20/run_${RUN}/${CPT}_eval_full_noise.log
