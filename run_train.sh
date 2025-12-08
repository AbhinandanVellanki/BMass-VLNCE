# unset DISPLAY
# export PYOPENGL_PLATFORM=egl
# export HABITAT_SIM_HEADLESS=1
# export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# # export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=/opt/conda/envs/vlnce/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}

# # python run.py --exp-config /home/ubuntu/BMass-VLNCE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml --run-type train

export IS_EVAL=0
#  xvfb-run -a -s "-screen 0 1024x768x24" \

# export EGL_DEVICE_ID=0
# export CUDA_VISIBLE_DEVICES=0

# export EGL_DEVICE_ID=0
# export CUDA_VISIBLE_DEVICES=0
# export MAGNUM_HEADLESS=ON
# export EGL_PLATFORM=device

# Check if nvidia EGL config exists, otherwise don't set it
# if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
#     export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# fi

xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
  --run-type train \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.noisy_features_path data/datasets/RxR_VLNCE_v0/noised_bert_features/rxr_{split}/{id:06}_{lang}_text_features.npz \
  TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.features_path  "data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz"
