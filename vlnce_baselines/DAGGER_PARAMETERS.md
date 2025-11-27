# DAgger Training Parameters

This document explains all configuration parameters for DAgger (Dataset Aggregation) training in VLN-CE.

## Core DAgger Algorithm (`IL.DAGGER`)

### `iterations`
- **Type**: Integer
- **Default**: 10
- **Description**: Number of DAgger iterations. Each iteration collects new on-policy data by rolling out the current policy, then trains for `epochs` on the aggregated dataset. More iterations = more data collection rounds.

### `update_size`
- **Type**: Integer
- **Default**: 5000
- **Description**: Number of episodes to collect per DAgger iteration. Each iteration rolls out the policy in the simulator and collects this many trajectories to add to the training dataset.

### `p`
- **Type**: Float (0.0 to 1.0)
- **Default**: 0.75
- **Description**: Probability of following the expert (oracle) policy during data collection. Computed as β = p^iteration.
  - `p=1.0` → Pure teacher forcing (always follow expert)
  - `p=0.75` → Exponential decay (starts at 75% expert, decreases each iteration)
  - `p=0.0` → Pure learner rollouts (never follow expert)
- **Formula**: At iteration `i`, the probability of following the expert is β = 0.75^i

### `expert_policy_sensor`
- **Type**: String
- **Default**: `SHORTEST_PATH_SENSOR`
- **Description**: The sensor type that provides oracle/expert actions during rollouts. This sensor must be registered in the Habitat task configuration.

### `expert_policy_sensor_uuid`
- **Type**: String
- **Default**: `shortest_path_sensor`
- **Description**: The UUID key to access expert actions from observations during training.

### `start_iteration`
- **Type**: Integer
- **Default**: 0
- **Description**: Starting iteration number. Useful for resuming training from a checkpoint.

## Storage & Performance (`IL.DAGGER`)

### `lmdb_features_dir`
- **Type**: String (file path)
- **Default**: `data/trajectories_dirs/debug/trajectories.lmdb`
- **Description**: Path where collected trajectory data is stored. DAgger saves observations and actions to this LMDB database to avoid re-simulating episodes every epoch.

### `lmdb_map_size`
- **Type**: Float (bytes)
- **Default**: 1.2e12 (1.2 TB)
- **Description**: Maximum size of the LMDB database. Must be large enough to hold all collected trajectories. LMDB pre-allocates this as virtual memory (doesn't consume actual disk until written).
- **Note**: Set this based on your expected dataset size. Each episode with RGB+Depth can be several MB.

### `lmdb_fp16`
- **Type**: Boolean
- **Default**: False
- **Description**: If True, stores observations in float16 instead of float32 to save ~50% disk space. Converted back to float32 when loading. May reduce precision slightly but generally safe for visual observations.

### `lmdb_commit_frequency`
- **Type**: Integer
- **Default**: 500
- **Description**: How often to commit writes to the LMDB database (every N episodes). 
  - Lower values = more frequent disk writes (safer but slower)
  - Higher values = faster but more data lost if crash occurs

### `preload_lmdb_features`
- **Type**: Boolean
- **Default**: False
- **Description**: If True, skips data collection and loads pre-collected trajectories from `lmdb_features_dir`. 
  - Set to **False** for initial training (collect new data)
  - Set to **True** to re-train on existing data without re-collecting

### `drop_existing_lmdb_features`
- **Type**: Boolean
- **Default**: True
- **Description**: If True, deletes existing LMDB database before starting training. Set False to append new data to existing database (advanced usage).

## Training Hyperparameters (`IL`)

### `batch_size`
- **Type**: Integer
- **Default**: 3-5 (varies by model)
- **Description**: Number of trajectories per training batch. DAgger groups trajectories of similar length for efficient batching. Larger batches = more stable gradients but higher memory usage.

### `epochs`
- **Type**: Integer
- **Default**: 4
- **Description**: Number of training epochs per DAgger iteration. After collecting `update_size` episodes, the policy trains for this many passes through the aggregated dataset.

### `inflection_weight_coef`
- **Type**: Float
- **Default**: 1.9
- **Description**: Weight multiplier for "inflection points" (timesteps when oracle action changes direction). If `use_iw=True`, loss at inflection steps is weighted by this coefficient to emphasize critical navigation decisions (e.g., turns, stops).

### `use_iw`
- **Type**: Boolean
- **Default**: True
- **Description**: Whether to use inflection weighting in the loss function. When enabled, timesteps at inflection points receive higher loss weights.

### `lr` (Learning Rate)
- **Type**: Float
- **Default**: 0.00025
- **Description**: Learning rate for the optimizer. Controls step size during gradient descent.

### `load_from_ckpt`
- **Type**: Boolean
- **Default**: False
- **Description**: Whether to initialize the policy from a pre-trained checkpoint before DAgger training.

### `ckpt_to_load`
- **Type**: String (file path)
- **Default**: `data/checkpoints/ckpt.0.pth`
- **Description**: Path to checkpoint file to load if `load_from_ckpt=True`. Useful for fine-tuning a pre-trained model with DAgger.

## Environment Setup

### `NUM_ENVIRONMENTS`
- **Type**: Integer
- **Default**: 1-8 (varies by configuration)
- **Description**: Number of parallel simulator environments for data collection. More environments = faster collection but higher memory/GPU usage. Each spawns a worker process.
- **Recommendation**: Use as many as your GPU memory allows (typically 4-8 per GPU).

### `SIMULATOR_GPU_IDS`
- **Type**: List of integers
- **Default**: `[0]`
- **Description**: GPU(s) to use for Habitat simulator rendering. With multiple environments, they can be distributed across these GPUs.
- **Example**: `[0, 1]` uses 2 GPUs for simulation

### `TORCH_GPU_ID`
- **Type**: Integer
- **Default**: 0
- **Description**: GPU for PyTorch model (policy network). Can differ from simulator GPUs to balance load.

### `ENV_NAME`
- **Type**: String
- **Default**: `VLNCEDaggerEnv`
- **Description**: Environment class name. Must be `VLNCEDaggerEnv` for DAgger training (provides expert actions).

## DAgger Training Flow

Here's how the parameters work together during training:

### Iteration 0:
1. **Data Collection**: Roll out policy in `NUM_ENVIRONMENTS` parallel sims
   - Collect `update_size` episodes (e.g., 5000)
   - Follow expert with probability β = p^0 = 0.75 (75% expert, 25% policy)
   - Save episodes to `lmdb_features_dir`

2. **Training**: Train policy on collected data
   - Train for `epochs` (e.g., 4) passes through the 5000 episodes
   - Batch size = `batch_size` (e.g., 3 trajectories per batch)
   - Apply inflection weighting if `use_iw=True`
   - Save checkpoints: `ckpt.0.pth`, `ckpt.1.pth`, `ckpt.2.pth`, `ckpt.3.pth`

### Iteration 1:
1. **Data Collection**: 
   - Collect 5000 new episodes with β = 0.75^1 = 0.56 (56% expert, 44% policy)
   - Append to LMDB (now 10,000 episodes total)

2. **Training**:
   - Train for 4 epochs on all 10,000 episodes
   - Save checkpoints: `ckpt.4.pth`, `ckpt.5.pth`, `ckpt.6.pth`, `ckpt.7.pth`

### Iterations 2-9:
- Continue pattern: collect 5000, train on all aggregated data
- β decreases exponentially: 0.42, 0.32, 0.24, 0.18, 0.13, 0.10, 0.075, 0.056
- Final dataset: 50,000 episodes
- Final checkpoint: `ckpt.39.pth` (10 iterations × 4 epochs - 1)

## Result

After 10 DAgger iterations, you have:
- **50,000 collected episodes** stored in LMDB
- **40 saved checkpoints** (one per epoch)
- A policy trained on progressively more of its own (less expert-guided) data
- The policy learns to correct its own mistakes through dataset aggregation

## Recommended Settings

### For Quick Experiments (Testing)
```yaml
IL:
  batch_size: 5
  epochs: 2
  DAGGER:
    iterations: 2
    update_size: 100
    p: 1.0  # Teacher forcing only
```

### For Full Training (Production)
```yaml
IL:
  batch_size: 3-5
  epochs: 4
  DAGGER:
    iterations: 10
    update_size: 5000
    p: 0.75  # Standard DAgger decay
```

### For Fine-tuning from Checkpoint
```yaml
IL:
  load_from_ckpt: True
  ckpt_to_load: data/checkpoints/pretrained_model.pth
  batch_size: 3
  epochs: 4
  DAGGER:
    iterations: 5
    update_size: 2000
    p: 0.5  # Faster decay for fine-tuning
```

## Memory Considerations

**LMDB Size Estimation**:
- RGB (224×224×3) + Depth (256×256×1) per observation ≈ 200 KB
- Average episode length: ~50 steps
- Storage per episode: ~10 MB
- 5,000 episodes ≈ 50 GB
- Set `lmdb_map_size` to 1.2e12 (1.2 TB) for safety

**GPU Memory**:
- Each environment requires ~2-4 GB GPU memory (simulator)
- Policy model: ~500 MB - 2 GB
- Batch processing: ~1-2 GB
- **Total**: 4-8 environments typically fit on a 32 GB GPU

## Troubleshooting

**"ConnectionResetError" during collection**:
- Reduce `NUM_ENVIRONMENTS` (try 1 first)
- Use `xvfb-run` for headless rendering
- Check GPU memory with `nvidia-smi`

**"LMDB map size exceeded"**:
- Increase `lmdb_map_size` (e.g., to 2.4e12)
- Enable `lmdb_fp16=True` to save space

**Training is slow**:
- Increase `NUM_ENVIRONMENTS` for faster collection
- Increase `batch_size` if GPU memory allows
- Use multiple GPUs via `SIMULATOR_GPU_IDS: [0, 1]`

## References

- **DAgger Paper**: Ross et al. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (2011)
- **VLN-CE**: Krantz et al. "Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments" (ECCV 2020)
