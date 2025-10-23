import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple

class BBBC006Loader(Dataset):
    """
    A PyTorch dataset for the BBBC006 (Human U2OS cells) focal stack dataset.
    This loader expects data to be pre-processed into .npz files using
    the preprocess_bbbc006.py script.
    """

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple = (512, 512),
        n_stack: int = 5,
        z_step_um: float = 2.0,
    ) -> None:
        super(BBBC006Loader, self).__init__()
        
        self.processed_dir = os.path.join(root_dir, "processed")
        self.processed_files = sorted(glob.glob(os.path.join(self.processed_dir, "*.npz")))
        
        if not self.processed_files:
            raise FileNotFoundError(f"No pre-processed .npz files found in {self.processed_dir}. "
                                    "Did you run the pre-processing script?")

        self.n_stack = n_stack
        self.total_slices = 32 # BBBC006 has 32 z-slices
        self.z_step_um = z_step_um
        self.stage = "train" # Default stage
        
        # --- MODIFIED ---
        # Grayscale images, but we will expand to 3 channels to match
        # the model's pre-trained input layers.
        self.mean_input = [0.5, 0.5, 0.5]
        self.std_input = [0.5, 0.5, 0.5]
        
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(self.mean_input, self.std_input)
        ])
        
        self.depth_transform = transforms.Resize(
            img_size, 
            interpolation=transforms.InterpolationMode.NEAREST, 
            antialias=False
        )
        self.to_tensor = transforms.ToTensor()

    def set_stage(self, stage: str):
        """Set the dataset stage ('train', 'val', or 'test')"""
        self.stage = stage

    def __len__(self) -> int:
        return len(self.processed_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Load the pre-processed data
        data = np.load(self.processed_files[index])
        full_stack = data['stack']  # [32, H, W]
        depth_map = data['depth']   # [H, W]
        aif_image = data['aif']     # [H, W]

        # Select n_stack slices
        if self.stage == 'train':
            # Randomly sample n_stack slices
            selected_indices = np.sort(random.sample(range(self.total_slices), self.n_stack))
        else:
            # Use a fixed, evenly-spaced selection for val/test
            selected_indices = np.linspace(0, self.total_slices - 1, self.n_stack, dtype=int)
        
        focal_stack = full_stack[selected_indices] # [n_stack, H, W]
        
        # Convert to Tensors
        # .unsqueeze(0) is not needed as ToTensor() handles [H, W] -> [1, H, W]
        aif_tensor = self.to_tensor(aif_image).float()       # [1, H, W]
        depth_tensor = self.to_tensor(depth_map).float()     # [1, H, W]
        
        # Convert stack: [n_stack, H, W] -> [n_stack, 1, H, W]
        stack_tensor = torch.stack([self.to_tensor(s) for s in focal_stack]).float() # [N, 1, H, W]

        # --- MODIFIED ---
        # Expand 1-channel grayscale to 3-channel "pseudo-RGB"
        # .expand() is memory-efficient as it just creates a view
        aif_tensor_rgb = aif_tensor.expand(3, -1, -1)             # [3, H, W]
        stack_tensor_rgb = stack_tensor.expand(-1, 3, -1, -1)     # [N, 3, H, W]

        # Apply transforms (Resize + Normalize)
        aif_transformed = self.img_transform(aif_tensor_rgb)
        stack_transformed = self.img_transform(stack_tensor_rgb)
        depth_transformed = self.depth_transform(depth_tensor)
        
        # Create focus distances tensor
        focus_distances = (selected_indices.astype(np.float32) + 1.0) * self.z_step_um
        focus_distances_tensor = torch.from_numpy(focus_distances)
        
        # Create a valid mask (all pixels are valid in this dataset)
        mask = torch.ones_like(depth_transformed, dtype=torch.bool)
        
        return aif_transformed, stack_transformed, depth_transformed, focus_distances_tensor, mask

