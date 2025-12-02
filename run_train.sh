# unset DISPLAY
# export PYOPENGL_PLATFORM=egl
# export HABITAT_SIM_HEADLESS=1
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/opt/conda/envs/vlnce/lib:$LD_LIBRARY_PATH
# python run.py --exp-config /home/ubuntu/BMass-VLNCE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml --run-type train
 xvfb-run -a -s "-screen 0 1024x768x24" \
 python run.py \
  --run-type train \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.noisy_features_path data/datasets/RxR_VLNCE_v0/noisy_text_features/rxr_{split}/{id:06}_{lang}_text_features.npz