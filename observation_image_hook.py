"""
Combined module for observation saving and noise injection
Handles both noise injection and saving of RGB/Depth observations during training
"""

import os
import cv2
import numpy as np
import torch


class ObservationNoiseInjector:
    """Add various types of noise to RGB and Depth observations"""
    
    def __init__(self, 
                 rgb_noise_type="gaussian", 
                 depth_noise_type="gaussian",
                 rgb_noise_params=None,
                 depth_noise_params=None):
        """
        Args:
            rgb_noise_type: Type of noise for RGB ("gaussian", "salt_pepper", "speckle", "motion_blur")
            depth_noise_type: Type of noise for depth ("gaussian", "dropout", "quantization")
            rgb_noise_params: Parameters for RGB noise
            depth_noise_params: Parameters for depth noise
        """
        self.rgb_noise_type = rgb_noise_type
        self.depth_noise_type = depth_noise_type
        
        # Default parameters
        self.rgb_noise_params = rgb_noise_params or {
            "gaussian": {"mean": 0, "std": 0.5},  # 50% noise
            "salt_pepper": {"amount": 0.5},
            "speckle": {"std": 0.5},
            "motion_blur": {"kernel_size": 15}
        }
        
        self.depth_noise_params = depth_noise_params or {
            "gaussian": {"mean": 0, "std": 0.5},  # 50% depth noise
            "dropout": {"dropout_prob": 0.5},  # 50% pixel dropout
            "quantization": {"num_levels": 10}  # More quantization
        }
        
        print(f"\n[NoiseInjector] Initialized")
        print(f"  RGB noise: {rgb_noise_type}")
        print(f"  Depth noise: {depth_noise_type}\n")
    
    def add_gaussian_noise(self, image, mean=0, std=0.02):
        """Add Gaussian noise to image"""
        noise = np.random.normal(mean, std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def add_salt_pepper_noise(self, image, amount=0.01):
        """Add salt and pepper noise"""
        noisy = image.copy()
        # Salt
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy[tuple(coords)] = 1.0
        # Pepper
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy[tuple(coords)] = 0.0
        return noisy
    
    def add_speckle_noise(self, image, std=0.02):
        """Add speckle noise"""
        noise = np.random.randn(*image.shape) * std
        noisy = image + image * noise
        return np.clip(noisy, 0, 1)
    
    def add_motion_blur(self, image, kernel_size=5):
        """Add motion blur"""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply to each channel
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:,:,i] = cv2.filter2D(image[:,:,i], -1, kernel)
            return blurred
        else:
            return cv2.filter2D(image, -1, kernel)
    
    def add_depth_dropout(self, depth, dropout_prob=0.05):
        """Randomly drop out depth pixels (simulate sensor failure)"""
        mask = np.random.rand(*depth.shape) > dropout_prob
        noisy_depth = depth * mask
        return noisy_depth
    
    def add_depth_quantization(self, depth, num_levels=50):
        """Quantize depth values (simulate lower precision sensor)"""
        max_depth = depth.max()
        if max_depth > 0:
            quantized = np.round(depth * num_levels / max_depth) * max_depth / num_levels
        else:
            quantized = depth
        return quantized
    
    def inject_rgb_noise(self, rgb):
        """Inject noise into RGB image"""
        # Normalize to [0, 1] if needed
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            rgb = rgb.astype(np.float32)
            was_uint8 = False
        
        # Apply noise based on type
        if self.rgb_noise_type == "gaussian":
            params = self.rgb_noise_params.get("gaussian", {})
            noisy_rgb = self.add_gaussian_noise(rgb, **params)
        elif self.rgb_noise_type == "salt_pepper":
            params = self.rgb_noise_params.get("salt_pepper", {})
            noisy_rgb = self.add_salt_pepper_noise(rgb, **params)
        elif self.rgb_noise_type == "speckle":
            params = self.rgb_noise_params.get("speckle", {})
            noisy_rgb = self.add_speckle_noise(rgb, **params)
        elif self.rgb_noise_type == "motion_blur":
            params = self.rgb_noise_params.get("motion_blur", {})
            noisy_rgb = self.add_motion_blur(rgb, **params)
        else:
            noisy_rgb = rgb
        
        # Convert back to uint8 if needed
        if was_uint8:
            noisy_rgb = (np.clip(noisy_rgb, 0, 1) * 255).astype(np.uint8)
        
        return noisy_rgb
    
    def inject_depth_noise(self, depth):
        """Inject noise into depth image"""
        depth = depth.astype(np.float32)
        
        # Apply noise based on type
        if self.depth_noise_type == "gaussian":
            params = self.depth_noise_params.get("gaussian", {})
            noisy_depth = self.add_gaussian_noise(depth, **params)
        elif self.depth_noise_type == "dropout":
            params = self.depth_noise_params.get("dropout", {})
            noisy_depth = self.add_depth_dropout(depth, **params)
        elif self.depth_noise_type == "quantization":
            params = self.depth_noise_params.get("quantization", {})
            noisy_depth = self.add_depth_quantization(depth, **params)
        else:
            noisy_depth = depth
        
        return noisy_depth
    
    def inject_noise(self, observations):
        """
        Inject noise into observations
        
        Args:
            observations: List of observation dicts or single observation dict
            
        Returns:
            Noisy observations in the same format
        """
        is_list = isinstance(observations, list)
        if not is_list:
            observations = [observations]
        
        noisy_observations = []
        
        for obs in observations:
            noisy_obs = obs.copy()
            
            # Add noise to RGB
            if "rgb" in obs:
                rgb = obs["rgb"]
                if torch.is_tensor(rgb):
                    rgb = rgb.cpu().numpy()
                
                noisy_rgb = self.inject_rgb_noise(rgb)
                
                # Convert back to tensor if needed - ENSURE FLOAT32
                if torch.is_tensor(obs["rgb"]):
                    noisy_obs["rgb"] = torch.from_numpy(noisy_rgb).float()  # Force float32
                else:
                    noisy_obs["rgb"] = noisy_rgb.astype(np.float32)  # Force float32
            
            # Add noise to depth
            if "depth" in obs:
                depth = obs["depth"]
                if torch.is_tensor(depth):
                    depth = depth.cpu().numpy()
                
                noisy_depth = self.inject_depth_noise(depth)
                
                # Convert back to tensor if needed - ENSURE FLOAT32
                if torch.is_tensor(obs["depth"]):
                    noisy_obs["depth"] = torch.from_numpy(noisy_depth).float()  # Force float32
                else:
                    noisy_obs["depth"] = noisy_depth.astype(np.float32)  # Force float32
            
            noisy_observations.append(noisy_obs)
        
        return noisy_observations if is_list else noisy_observations[0]


class ObservationNoiseInjectorPatch:
    """Add random patches/occlusions to RGB observations"""
    
    def __init__(self, 
                 num_patches=5,
                 patch_size_range=(10, 50),
                 patch_type="random",
                 patch_color=None):
        """
        Args:
            num_patches: Number of patches to add per image
            patch_size_range: (min, max) tuple for patch size in pixels
            patch_type: Type of patch - "random" (random color), "black", "white", "gray"
            patch_color: Specific RGB color for patches (overrides patch_type) e.g., [255, 0, 0] for red
        """
        self.num_patches = num_patches
        self.patch_size_range = patch_size_range
        self.patch_type = patch_type
        self.patch_color = patch_color
        
        print(f"\n[PatchNoiseInjector] Initialized")
        print(f"  Num patches: {num_patches}")
        print(f"  Patch size range: {patch_size_range}")
        print(f"  Patch type: {patch_type}")
        if patch_color is not None:
            print(f"  Patch color: {patch_color}")
        print()
    
    def add_patches(self, rgb):
        """Add random patches to RGB image"""
        # Ensure image is in correct format
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            rgb = rgb.astype(np.float32)
            was_uint8 = False
        
        h, w = rgb.shape[:2]
        patched_rgb = rgb.copy()
        
        for _ in range(self.num_patches):
            # Random patch size
            patch_h = np.random.randint(self.patch_size_range[0], self.patch_size_range[1])
            patch_w = np.random.randint(self.patch_size_range[0], self.patch_size_range[1])
            
            # Random position (ensure patch fits in image)
            if h > patch_h and w > patch_w:
                y = np.random.randint(0, h - patch_h)
                x = np.random.randint(0, w - patch_w)
                
                # Determine patch color
                if self.patch_color is not None:
                    # Use specified color (normalize to [0, 1])
                    color = np.array(self.patch_color, dtype=np.float32) / 255.0
                elif self.patch_type == "black":
                    color = np.array([0.0, 0.0, 0.0])
                elif self.patch_type == "white":
                    color = np.array([1.0, 1.0, 1.0])
                elif self.patch_type == "gray":
                    gray_val = np.random.uniform(0.3, 0.7)
                    color = np.array([gray_val, gray_val, gray_val])
                else:  # random
                    color = np.random.rand(3)
                
                # Apply patch
                patched_rgb[y:y+patch_h, x:x+patch_w] = color
        
        # Convert back to uint8 if needed
        if was_uint8:
            patched_rgb = (np.clip(patched_rgb, 0, 1) * 255).astype(np.uint8)
        
        return patched_rgb
    
    def inject_noise(self, observations):
        """
        Inject patch noise into RGB observations
        
        Args:
            observations: List of observation dicts or single observation dict
            
        Returns:
            Noisy observations with patches
        """
        is_list = isinstance(observations, list)
        if not is_list:
            observations = [observations]
        
        noisy_observations = []
        
        for obs in observations:
            noisy_obs = obs.copy()
            
            # Add patches to RGB only
            if "rgb" in obs:
                rgb = obs["rgb"]
                if torch.is_tensor(rgb):
                    rgb = rgb.cpu().numpy()
                
                patched_rgb = self.add_patches(rgb)
                
                # Convert back to tensor if needed - ENSURE FLOAT32
                if torch.is_tensor(obs["rgb"]):
                    noisy_obs["rgb"] = torch.from_numpy(patched_rgb).float()
                else:
                    noisy_obs["rgb"] = patched_rgb.astype(np.float32)
            
            # Keep depth unchanged
            if "depth" in obs:
                noisy_obs["depth"] = obs["depth"]
            
            noisy_observations.append(noisy_obs)
        
        return noisy_observations if is_list else noisy_observations[0]


class ObservationSaver:
    """Save RGB and Depth observations to disk during training"""
    
    def __init__(self, save_dir="training_observations", save_frequency=50, save_noisy=True):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.save_noisy = save_noisy
        self.step_count = 0
        
        # Create directories for original images
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
        
        # Create directories for noisy images
        if save_noisy:
            os.makedirs(os.path.join(save_dir, "rgb_noisy"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "depth_noisy"), exist_ok=True)
        
        print(f"\n[ObservationSaver] Saving observations to: {os.path.abspath(save_dir)}")
        print(f"[ObservationSaver] Save frequency: every {save_frequency} steps")
        print(f"[ObservationSaver] Save noisy images: {save_noisy}\n")
    
    def save(self, observations, episode_ids, noisy_observations=None):
        """Save RGB and depth from observations"""
        self.step_count += 1
        
        if self.step_count % self.save_frequency != 0:
            return
        
        # observations is a list from multiple environments
        for env_idx, obs in enumerate(observations):
            ep_id = episode_ids[env_idx] if env_idx < len(episode_ids) else f"env{env_idx}"
            
            # Save original RGB
            if "rgb" in obs:
                rgb = obs["rgb"]
                if hasattr(rgb, 'cpu'):  # torch tensor
                    rgb = rgb.cpu().numpy()
                
                # Ensure uint8
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                
                # Save as image
                rgb_path = os.path.join(
                    self.save_dir, "rgb", 
                    f"step_{self.step_count:06d}_env{env_idx}_ep{ep_id}.jpg"
                )
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Save original Depth
            if "depth" in obs:
                depth = obs["depth"]
                if hasattr(depth, 'cpu'):  # torch tensor
                    depth = depth.cpu().numpy()
                
                depth = depth.squeeze()
                
                # Normalize to 0-255 for visualization
                depth_vis = np.clip(depth * 255 / 10.0, 0, 255).astype(np.uint8)
                
                depth_path = os.path.join(
                    self.save_dir, "depth",
                    f"step_{self.step_count:06d}_env{env_idx}_ep{ep_id}.jpg"
                )
                cv2.imwrite(depth_path, depth_vis)
            
            # Save noisy versions if provided
            if self.save_noisy and noisy_observations is not None:
                noisy_obs = noisy_observations[env_idx]
                
                # Save noisy RGB
                if "rgb" in noisy_obs:
                    noisy_rgb = noisy_obs["rgb"]
                    if hasattr(noisy_rgb, 'cpu'):
                        noisy_rgb = noisy_rgb.cpu().numpy()
                    
                    if noisy_rgb.max() <= 1.0:
                        noisy_rgb = (noisy_rgb * 255).astype(np.uint8)
                    else:
                        noisy_rgb = noisy_rgb.astype(np.uint8)
                    
                    noisy_rgb_path = os.path.join(
                        self.save_dir, "rgb_noisy",
                        f"step_{self.step_count:06d}_env{env_idx}_ep{ep_id}.jpg"
                    )
                    cv2.imwrite(noisy_rgb_path, cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR))
                
                # Save noisy Depth
                if "depth" in noisy_obs:
                    noisy_depth = noisy_obs["depth"]
                    if hasattr(noisy_depth, 'cpu'):
                        noisy_depth = noisy_depth.cpu().numpy()
                    
                    noisy_depth = noisy_depth.squeeze()
                    noisy_depth_vis = np.clip(noisy_depth * 255 / 10.0, 0, 255).astype(np.uint8)
                    
                    noisy_depth_path = os.path.join(
                        self.save_dir, "depth_noisy",
                        f"step_{self.step_count:06d}_env{env_idx}_ep{ep_id}.jpg"
                    )
                    cv2.imwrite(noisy_depth_path, noisy_depth_vis)
        
        if self.step_count % (self.save_frequency * 10) == 0:
            print(f"[ObservationSaver] Saved observations at step {self.step_count}")

