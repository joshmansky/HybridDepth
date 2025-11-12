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
    
    Each .npz file should contain:
    - 'full_stack': (Z, H, W, 3) np.ndarray
    - 'depth_map': (H, W, 3) np.ndarray
    - 'z_step': float scalar
    - 'total_slices': int scalar (Z)
    """

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple = (256, 256),
        n_stack: int = 10,
        n_buckets: int = 5,
    ) -> None:
        super(AllenCellLoader, self).__init__()
        
        self.processed_dir = os.path.join(root_dir, "processed")
        self.processed_files = sorted(glob.glob(os.path.join(self.processed_dir, "*.npz")))
        
        if not self.processed_files:
            raise FileNotFoundError(f"No pre-processed .npz files found in {self.processed_dir}. "
                                    "Did you run the pre-processing script?")

        self.n_stack = n_stack
        self.n_buckets = n_buckets
        self.samples_per_bucket = n_stack // n_buckets
        
        if n_stack % n_buckets != 0:
            raise ValueError(f"n_stack ({n_stack}) must be divisible by n_buckets ({n_buckets}).")
            
        self.stage = "train" # Default stage
        
        # 3 channels (DNA, Mem, Struct)
        # These are dummy values; you should calculate and use real ones
        self.mean_input = [0.5, 0.5, 0.5]
        self.std_input = [0.5, 0.5, 0.5]
        
        # Note: ToTensor() converts (H, W, C) [0-255] to (C, H, W) [0.0-1.0]
        # Our data is already (H, W, C) float, but ToTensor will handle the permute
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
        """Set the dataset stage ('train', 'val', or 'test')"""
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
                selected_indices.extend(random.sample(list(bucket), self.samples_per_bucket))
            selected_indices = np.array(selected_indices)
        else:
            # Use a fixed, evenly-spaced selection for val/test
            selected_indices = np.linspace(0, total_slices - 1, self.n_stack, dtype=int)
        
        return np.sort(selected_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Load the pre-processed data
        data = np.load(self.processed_files[index])
        full_stack = data['full_stack']  # (Z, H, W, 3)
        depth_map = data['depth_map']   # (H, W, 3)
        z_step = data['z_step']
        total_slices = data['total_slices']

        # 1. Select n_stack slices
        selected_indices = self._get_slice_indices(total_slices)
        focal_stack = full_stack[selected_indices] # (n_stack, H, W, 3)
        
        # 2. Convert to Tensors and permute C, H, W
        # focal_stack: (n_stack, H, W, 3) -> (n_stack, 3, H, W)
        # ToTensor expects (H, W, C), so we loop.
        stack_tensor_list = [self.to_tensor(s.astype(np.float32) / 65535.0) for s in focal_stack]
        stack_tensor = torch.stack(stack_tensor_list).float() # (n_stack, 3, H, W)

        # depth_map: (H, W, 3) -> (3, H, W)
        depth_tensor = self.to_tensor(depth_map).float() # (3, H, W)
        
        # 3. Apply transforms (Resize + Normalize)
        stack_transformed = self.img_transform(stack_tensor)
        depth_transformed = self.depth_transform(depth_tensor)
        
        # 4. Create focus distances tensor
        # (index + 1) * z_step to match BBBC006 logic (positive distances)
        focus_distances = (selected_indices.astype(np.float32) + 1.0) * z_step
        focus_distances_tensor = torch.from_numpy(focus_distances)
        
        # 5. Create a valid mask
        # We assume all pixels in the depth map are valid.
        # We only need the (1, H, W) mask for the loss function.
        # We can take the mask from the first channel (DNA).
        mask = (depth_transformed[0:1, :, :] > 0).bool()
        
        # NOTE: Returning 4 items. Your model may expect 5 (with AIF).
        # If so, you may need to compute an AIF image.
        # Return: stack, depth, focus_distances, mask
        return stack_transformed, depth_transformed, focus_distances_tensor, mask