import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List


class AllenCellLoader(Dataset):
    """
    A PyTorch dataset for the pre-processed Allen Cell focal stack dataset.
    This loader expects data to be pre-processed into .npz files using
    the preprocess_allencell.py script.
    
    This version loads the 3-channel data but selects ONE channel 
    (e.g., 'dna') and expands it to 3 channels to mimic datasets 
    like BBBC006. It returns a 1-channel depth map.
    """

    def __init__(
        self,
        processed_dir: str, 
        img_size: Tuple = (256, 256),
        n_stack: int = 10,
        n_buckets: int = 5,
        channel_to_use: str = 'dna', # <-- NEW: 'dna', 'membrane', or 'structure'
    ) -> None:
        super(AllenCellLoader, self).__init__()
        
        self.processed_dir = processed_dir
        self.processed_files = sorted(glob.glob(os.path.join(self.processed_dir, "*.npz")))
        
        if not self.processed_files:
            raise FileNotFoundError(f"No pre-processed .npz files found in {self.processed_dir}. "
                                    "Did you mount your Google Drive and check the path?")

        self.n_stack = n_stack
        self.n_buckets = n_buckets
        self.samples_per_bucket = n_stack // n_buckets
        
        if n_stack % n_buckets != 0:
            raise ValueError(f"n_stack ({n_stack}) must be divisible by n_buckets ({n_buckets}).")
        
        # --- NEW: Map channel name to its index in the .npz file ---
        self.channel_map = {'dna': 0, 'membrane': 1, 'structure': 2}
        if channel_to_use not in self.channel_map:
            raise ValueError(f"channel_to_use must be one of {list(self.channel_map.keys())}")
        self.channel_idx = self.channel_map[channel_to_use]
        print(f"AllenCellLoader initialized. Using channel: {channel_to_use} (index {self.channel_idx})")
        # --- END NEW ---
            
        self.stage = "train" # Default stage
        
        # These are dummy values; you should calculate and use real ones
        self.mean_input = [0.5, 0.5, 0.5]
        self.std_input = [0.5, 0.5, 0.5]
        
        self.to_tensor = transforms.ToTensor()

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(self.mean_input, self.std_input)
        ])
        
        # Depth map must use NEAREST interpolation
        self.depth_transform = transforms.Resize(
            img_size, 
            interpolation=transforms.InterpolationMode.NEAREST, 
            antialias=False
        )

    def set_stage(self, stage: str):
        self.stage = stage

    def __len__(self) -> int:
        return len(self.processed_files)

    def _get_slice_indices(self, total_slices: int) -> np.ndarray:
        """
        Gets n_stack indices, either randomly (train) or fixed (val/test).
        Uses the "5 buckets, 2 samples" logic.
        """
        if self.stage == 'train':
            # Create 5 buckets (quintiles)
            buckets = np.array_split(range(total_slices), self.n_buckets)
            selected_indices = []
            for bucket in buckets:
                # Randomly sample 2 indices from each bucket
                if len(bucket) < self.samples_per_bucket:
                    # Handle rare case where bucket is smaller than samples
                    selected_indices.extend(np.random.choice(bucket, self.samples_per_bucket, replace=True))
                else:
                    selected_indices.extend(random.sample(list(bucket), self.samples_per_bucket))
            selected_indices = np.array(selected_indices)
        else:
            # Use a fixed, evenly-spaced selection for val/test
            selected_indices = np.linspace(0, total_slices - 1, self.n_stack, dtype=int)
        
        return np.sort(selected_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Load the pre-processed data
        data = np.load(self.processed_files[index])
        
        # Load as float32; uint16 is not supported by torch.stack
        full_stack_3channel = data['full_stack'].astype(np.float32)  # (Z, H, W, 3)
        depth_map_3channel = data['depth_map']   # (H, W, 3)
        z_step = data['z_step']
        total_slices = data['total_slices']

        # --- NEW: Select only the desired channel ---
        stack_1channel_3d = full_stack_3channel[..., self.channel_idx] # (Z, H, W)
        depth_1channel_2d = depth_map_3channel[..., self.channel_idx] # (H, W)
        # --- END NEW ---

        # --- Create an All-in-Focus (AIF) image from the selected channel ---
        aif_image_1channel = np.max(stack_1channel_3d, axis=0) # (H, W)
        
        # Convert AIF to tensor (H, W) -> (1, H, W)
        aif_tensor = self.to_tensor(aif_image_1channel / 65535.0).float()
        
        # --- NEW: Expand 1-channel AIF to 3-channel "pseudo-RGB" ---
        aif_tensor = aif_tensor.expand(3, -1, -1) # (3, H, W)
        
        # Apply transforms (Resize + Normalize)
        aif_transformed = self.img_transform(aif_tensor)
        # --- END AIF ---

        # 1. Select n_stack slices
        selected_indices = self._get_slice_indices(total_slices)
        focal_stack_1channel = stack_1channel_3d[selected_indices] # (n_stack, H, W)
        
        # 2. Convert to Tensors and permute C, H, W
        # focal_stack: (n_stack, H, W) -> (n_stack, 1, H, W)
        stack_tensor_list = [self.to_tensor(s / 65535.0) for s in focal_stack_1channel]
        stack_tensor = torch.stack(stack_tensor_list).float() # (n_stack, 1, H, W)

        # --- NEW: Expand 1-channel stack to 3-channel "pseudo-RGB" ---
        stack_tensor = stack_tensor.expand(-1, 3, -1, -1) # (n_stack, 3, H, W)
        
        # 3. Process 1-Channel Depth Map
        # Convert depth_map: (H, W) -> (1, H, W)
        depth_tensor = self.to_tensor(depth_1channel_2d).float() # (1, H, W)
        
        # 4. Apply transforms (Resize + Normalize)
        stack_transformed = self.img_transform(stack_tensor)
        depth_transformed = self.depth_transform(depth_tensor) # This is (1, H, W)
        
        # 5. Create focus distances tensor
        focus_distances = (selected_indices.astype(np.float32) + 1.0) * z_step
        focus_distances_tensor = torch.from_numpy(focus_distances)
        
        # 6. Create a valid mask
        mask = (depth_transformed > 0).bool() # (1, H, W)
        
        # Return 5 items in the correct order
        return aif_transformed, stack_transformed, depth_transformed, focus_distances_tensor, mask