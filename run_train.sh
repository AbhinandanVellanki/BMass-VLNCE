unset DISPLAY
# export PYOPENGL_PLATFORM=egl
# export HABITAT_SIM_HEADLESS=1
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/opt/conda/envs/vlnce/lib:$LD_LIBRARY_PATH
# python run.py --exp-config /home/ubuntu/BMass-VLNCE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml --run-type train
python run.py \
  --run-type train \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.noisy_features_path data/datasets/RxR_VLNCE_v0/noisy_text_features/rxr_{split}/{id:06}_{lang}_text_features.npz \
  IL.batch_size 1 \
  NUM_ENVIRONMENTS 1 \
  IL.epochs 1 \
  IL.RECOLLECT_TRAINER.preload_size 5
