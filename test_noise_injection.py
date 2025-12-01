#!/usr/bin/env python3
"""
Test script to verify noise injection integration with recollect_trainer
"""

import numpy as np
import torch
from observation_image_hook import (
    ObservationNoiseInjector,
    ObservationNoiseInjectorPatch,
)

def test_gaussian_noise():
    """Test Gaussian noise injection"""
    print("\n=== Testing Gaussian Noise ===")
    
    # Create sample observations
    rgb = np.random.rand(3, 224, 224).astype(np.float32)  # (C, H, W)
    depth = np.random.rand(1, 224, 224).astype(np.float32)
    
    obs = {
        'rgb': torch.from_numpy(rgb),
        'depth': torch.from_numpy(depth)
    }
    
    # Initialize noise injector
    injector = ObservationNoiseInjector(
        rgb_noise_type='gaussian',
        depth_noise_type='gaussian',
        rgb_noise_params={'gaussian': {'mean': 0, 'std': 0.1}},
        depth_noise_params={'gaussian': {'mean': 0, 'std': 0.1}}
    )
    
    # Inject noise
    noisy_obs = injector.inject_noise(obs)
    
    print(f"Original RGB range: [{obs['rgb'].min():.3f}, {obs['rgb'].max():.3f}]")
    print(f"Noisy RGB range: [{noisy_obs['rgb'].min():.3f}, {noisy_obs['rgb'].max():.3f}]")
    print(f"Original Depth range: [{obs['depth'].min():.3f}, {obs['depth'].max():.3f}]")
    print(f"Noisy Depth range: [{noisy_obs['depth'].min():.3f}, {noisy_obs['depth'].max():.3f}]")
    print(f"RGB difference (mean): {torch.abs(obs['rgb'] - noisy_obs['rgb']).mean():.3f}")
    print("✓ Gaussian noise test passed!")

def test_patch_noise():
    """Test patch noise injection"""
    print("\n=== Testing Patch Noise ===")
    
    # Create sample observations
    rgb = np.random.rand(3, 224, 224).astype(np.float32)
    
    obs = {
        'rgb': torch.from_numpy(rgb)
    }
    
    # Initialize patch noise injector
    injector = ObservationNoiseInjectorPatch(
        num_patches=5,
        patch_size_range=(20, 50),
        patch_type='random'
    )
    
    # Inject noise
    noisy_obs = injector.inject_noise(obs)
    
    print(f"Original RGB range: [{obs['rgb'].min():.3f}, {obs['rgb'].max():.3f}]")
    print(f"Noisy RGB range: [{noisy_obs['rgb'].min():.3f}, {noisy_obs['rgb'].max():.3f}]")
    print(f"Pixels changed: {(obs['rgb'] != noisy_obs['rgb']).sum().item()} / {obs['rgb'].numel()}")
    print("✓ Patch noise test passed!")

def test_batch_processing():
    """Test batch processing (as used in trainer)"""
    print("\n=== Testing Batch Processing ===")
    
    batch_size = 4
    
    # Create batch observations
    observations_batch = {
        'rgb': torch.rand(batch_size, 3, 224, 224),
        'depth': torch.rand(batch_size, 1, 224, 224)
    }
    
    print(f"Batch shape: RGB={observations_batch['rgb'].shape}, Depth={observations_batch['depth'].shape}")
    
    # Initialize noise injector
    injector = ObservationNoiseInjector(
        rgb_noise_type='gaussian',
        depth_noise_type='gaussian',
        rgb_noise_params={'gaussian': {'mean': 0, 'std': 0.05}},
        depth_noise_params={'gaussian': {'mean': 0, 'std': 0.05}}
    )
    
    # Convert batch to list (as done in trainer)
    obs_list = []
    for i in range(batch_size):
        obs = {k: v[i] for k, v in observations_batch.items()}
        obs_list.append(obs)
    
    # Inject noise
    noisy_obs_list = injector.inject_noise(obs_list)
    
    # Convert back to batch
    noisy_batch = {}
    for key in obs_list[0].keys():
        noisy_batch[key] = torch.stack([obs[key] for obs in noisy_obs_list])
    
    print(f"Noisy batch shape: RGB={noisy_batch['rgb'].shape}, Depth={noisy_batch['depth'].shape}")
    print(f"RGB difference (mean): {torch.abs(observations_batch['rgb'] - noisy_batch['rgb']).mean():.3f}")
    print(f"Depth difference (mean): {torch.abs(observations_batch['depth'] - noisy_batch['depth']).mean():.3f}")
    print("✓ Batch processing test passed!")

def test_all_noise_types():
    """Test all available noise types"""
    print("\n=== Testing All Noise Types ===")
    
    rgb = np.random.rand(3, 224, 224).astype(np.float32)
    depth = np.random.rand(1, 224, 224).astype(np.float32)
    
    obs = {
        'rgb': torch.from_numpy(rgb),
        'depth': torch.from_numpy(depth)
    }
    
    noise_types = [
        ('gaussian', 'gaussian'),
        ('salt_pepper', 'dropout'),
        ('speckle', 'quantization'),
        ('motion_blur', 'gaussian'),
    ]
    
    for rgb_noise, depth_noise in noise_types:
        injector = ObservationNoiseInjector(
            rgb_noise_type=rgb_noise,
            depth_noise_type=depth_noise
        )
        noisy_obs = injector.inject_noise(obs.copy())
        print(f"  ✓ {rgb_noise}/{depth_noise} noise applied successfully")
    
    print("✓ All noise types test passed!")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Noise Injection Integration")
    print("=" * 60)
    
    test_gaussian_noise()
    test_patch_noise()
    test_batch_processing()
    test_all_noise_types()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou can now use noise injection in training by:")
    print("1. Adding USE_NOISE_INJECTION: True to your config")
    print("2. Configuring noise parameters (see noise_config_example.yaml)")
    print("3. Running your training as usual")

