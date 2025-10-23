import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from model.main import DepthNetModule  # Import the main model class
from dataloader.dataset import BBBC006DataModule  # Import your datamodule

# --- Configuration ---
CHECKPOINT_PATH = "/content/HybridDepth/checkpoints_bbbc006/best_37.ckpt"
BBBC006_ROOT = "/content/HybridDepth/dataset/BBBC006"
SAMPLE_INDEX = random.randint(0, 70) # Pick a random sample (0-76, as 10% of 768 is ~77)

# --- NEW: Define the absolute, known depth range of the dataset ---
# From our preprocessing: (index + 1) * z_step_um
# Min: (0 + 1) * 2.0 = 2.0
# Max: (31 + 1) * 2.0 = 64.0
ABSOLUTE_VMIN = 2.0
ABSOLUTE_VMAX = 64.0
# ------------------------------------------------------------------

def unnormalize_image(tensor):
    """Un-normalizes a tensor from [0.5, 0.5, 0.5] mean/std"""
    # (img * std) + mean
    tensor = tensor.clone() * 0.5 + 0.5
    # Permute from (C, H, W) to (H, W, C) for plotting
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    # Clip to [0, 1] range
    return np.clip(img_np, 0, 1)

def main():
    print(f"Loading sample {SAMPLE_INDEX} for visualization...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    try:
        model = DepthNetModule.load_from_checkpoint(CHECKPOINT_PATH)
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("✅ Model loaded successfully from checkpoint.")
    except FileNotFoundError:
        print(f"❌ ERROR: Checkpoint file not found at {CHECKPOINT_PATH}")
        return
    except Exception as e:
        print(f"❌ ERROR: Could not load model. {e}")
        print("   Make sure all model class (DepthNetModule, etc.) are defined.")
        return

    # 2. Load the dataset
    # We use the same settings as in evaluation to get the *exact* same test set
    datamodule = BBBC006DataModule(
        bbbc006_data_root=BBBC006_ROOT,
        img_size=[512, 512],
        batch_size=1,
        num_workers=0,
        n_stack=5,
        val_split_percent=0.1,
        test_split_percent=0.1,
        z_step_um=2.0,
        seed=42
    )
    datamodule.setup(stage='test') # Call setup to create the test_dataset
    test_dataset = datamodule.test_dataset
    print(f"✅ Test dataset loaded. Total size: {len(test_dataset)} samples.")

    # 3. Get one data sample
    # The loader returns: aif_transformed, stack_transformed, depth_transformed, focus_distances_tensor, mask
    try:
        aif_img, focal_stack, depth_gt, foc_dist, mask = test_dataset[SAMPLE_INDEX]
    except IndexError:
        print(f"❌ ERROR: Sample index {SAMPLE_INDEX} is out of bounds for test set (size {len(test_dataset)}).")
        print("   Try a smaller index (e.g., 0).")
        return

    # 4. Prepare the sample for the model
    # We need to add a batch dimension (B=1) and send to the device
    aif_img_batch = aif_img.unsqueeze(0).to(device)
    focal_stack_batch = focal_stack.unsqueeze(0).to(device)
    foc_dist_batch = foc_dist.unsqueeze(0).to(device)

    # 5. Run inference
    print("Running model inference...")
    with torch.no_grad(): # Disable gradient calculation
        # The model's forward pass (from main.py) returns multiple values
        depth_prd, rel_depth, gl_depth, _, scale_map = model.forward(
            aif_img_batch, focal_stack_batch, foc_dist_batch
        )
    print("✅ Inference complete.")

    # 6. Process outputs for visualization
    # Move to CPU, convert to NumPy, remove batch/channel dims
    pred_np = depth_prd.squeeze().cpu().numpy()
    gt_np = depth_gt.squeeze().cpu().numpy()
    
    # Get a few slices from the input stack for plotting
    n_stack = focal_stack.shape[0]
    slice_1 = unnormalize_image(focal_stack[0])
    slice_mid = unnormalize_image(focal_stack[n_stack // 2])
    slice_last = unnormalize_image(focal_stack[-1])

    # 7. Plot the results
    print("Plotting results...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Since we used pseudo-RGB, we can just show one channel (e.g., [:, :, 0])
    # or let imshow handle it (it will look gray)
    axes[0].imshow(slice_1)
    axes[0].set_title(f"Input Slice 1 (focus={foc_dist[0].item():.1f}µm)")
    axes[0].axis('off')
    
    axes[1].imshow(slice_mid)
    axes[1].set_title(f"Input Slice {n_stack // 2} (focus={foc_dist[n_stack // 2].item():.1f}µm)")
    axes[1].axis('off')

    axes[2].imshow(slice_last)
    axes[2].set_title(f"Input Slice {n_stack} (focus={foc_dist[-1].item():.1f}µm)")
    axes[2].axis('off')

    # --- MODIFIED SECTION ---
    # Use the ABSOLUTE known data range for vmin and vmax
    # This gives a true picture of the model's absolute prediction.
    vmin = ABSOLUTE_VMIN
    vmax = ABSOLUTE_VMAX
    # ------------------------
    
    im_gt = axes[3].imshow(gt_np, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[3].set_title("Ground Truth Depth")
    axes[3].axis('off')
    
    im_pred = axes[4].imshow(pred_np, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[4].set_title("Predicted Depth")
    axes[4].axis('off')

    # Add a colorbar
    # We still use one of the images (im_pred) to create the colorbar,
    # but the scale it shows is now fixed to [vmin, vmax]
    fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.75, label="Depth (µm)")
    
    plt.suptitle(f"Inference for Test Sample {SAMPLE_INDEX}", fontsize=16)
    plt.tight_layout()
    
    # Save or show the plot
    plt.savefig("inference_visualization.png")
    print("✅ Visualization saved to 'inference_visualization.png'")
    plt.show()


if __name__ == "__main__":
    main()

