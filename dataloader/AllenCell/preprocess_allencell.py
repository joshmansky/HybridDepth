import os
import quilt3
import pandas as pd
import numpy as np
from aicsimageio import AICSImage
from tqdm import tqdm
import warnings

# ==============================================================================
#   Configuration
# ==============================================================================

# --- Set this to the number of cells you want in your local dataset ---
NUM_CELLS_TO_PROCESS = 1000  # Start with 1k-5k, you can increase later
# ----------------------------------------------------------------------

# --- Set this to your desired root directory ---
# We will create /content/HybridDepth/AllenCell/processed
DATASET_ROOT = "/content/HybridDepth/AllenCell"
# -------------------------------------------------

PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
#   Helper Functions for Processing
# ==============================================================================

def get_channel_indices(name_dict, group_key, channel_names):
    """
    Gets the integer indices for a list of channel names.
    
    Args:
        name_dict (dict): The 'name_dict' from the manifest.
        group_key (str): 'crop_raw' or 'crop_seg'.
        channel_names (list[str]): List of names to find, e.g., ['dna', 'membrane'].
        
    Returns:
        list[int]: List of corresponding integer indices.
    """
    channel_list = name_dict[group_key]
    indices = [channel_list.index(name) for name in channel_names]
    return indices

def create_depth_map(mask_3d, z_scale):
    """
    Creates a 2D depth map in microns from a 3D boolean mask.
    The background (where there is no mask) is 0.
    """
    if not mask_3d.any():
        # Return an empty map if mask is all False
        return np.zeros(mask_3d.shape[1:], dtype=np.float32)
    
    # Get the Z-index of the highest point (first 'True' from the top)
    depth_indices = np.argmax(mask_3d, axis=0)
    
    # Create a 2D mask to find where *any* cell part exists
    any_mask_in_column = mask_3d.any(axis=0)
    
    # Set background pixels (where no mask exists) to 0
    depth_indices = np.where(any_mask_in_column, depth_indices, 0)
    
    # Convert Z-index to physical depth in microns
    depth_microns = depth_indices.astype(np.float32) * z_scale
    return depth_microns

# ==============================================================================
#   Main Preprocessing Script
# ==============================================================================

def main():
    # Suppress warnings from aicsimageio
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Accessing Quilt data package...")
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")

    print("Fetching and loading the metadata.csv file...")
    try:
        manifest_df = pkg['metadata.csv']['metadata.csv'].load()
        print(f"Successfully loaded metadata for {len(manifest_df)} cells.")
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return

    print(f"--- Starting preprocessing for {NUM_CELLS_TO_PROCESS} cells ---")
    print(f"Saving processed files to: {PROCESSED_DIR}")

    # Use tqdm for a nice progress bar
    for index, row in tqdm(manifest_df.head(NUM_CELLS_TO_PROCESS).iterrows(), total=NUM_CELLS_TO_PROCESS):
        
        cell_id = row['CellId']
        save_path = os.path.join(PROCESSED_DIR, f"cell_{cell_id}.npz")
        
        # Skip if this file has already been processed
        if os.path.exists(save_path):
            continue
            
        try:
            # ----------------------------------------------------------
            # 1. Load Metadata and Z-Scale
            # ----------------------------------------------------------
            name_dict = eval(row['name_dict'])
            scale_list = eval(row['scale_micron'])
            z_scale = scale_list[0] # Microns per Z-slice
            
            # ----------------------------------------------------------
            # 2. Load 3-Channel Input Stack (X)
            # ----------------------------------------------------------
            raw_image_pkg_path = row['crop_raw']
            
            with pkg[raw_image_pkg_path].open() as f:
                img_reader = AICSImage(f)
                
                # Get indices for [dna, membrane, structure]
                raw_indices = get_channel_indices(name_dict, 'crop_raw', ['dna', 'membrane', 'structure'])
                
                # Load all channels into a (Z, C, Y, X) array
                # Note: C order is [dna, mem, struct]
                full_raw_stack_czyx = img_reader.get_image_data("ZCYX", T=0, C=raw_indices)
            
            # Permute to (Z, Y, X, C) which is easier to sample from
            full_raw_stack = np.transpose(full_raw_stack_czyx, (0, 2, 3, 1)) # (Z, Y, X, C)
            
            total_slices = full_raw_stack.shape[0]

            # ----------------------------------------------------------
            # 3. Load 3-Channel Segmentation Masks and create Depth Maps (Y)
            # ----------------------------------------------------------
            seg_image_pkg_path = row['crop_seg']
            
            with pkg[seg_image_pkg_path].open() as f:
                seg_reader = AICSImage(f)
                
                # Get indices for [dna_seg, mem_seg, struct_seg]
                seg_indices = get_channel_indices(name_dict, 'crop_seg', 
                                  ['dna_segmentation', 'membrane_segmentation', 'struct_segmentation'])
                
                # Load all 3 masks: (Z, C, Y, X)
                # Note: C order is [dna, mem, struct]
                full_seg_masks_zcyx = seg_reader.get_image_data("ZCYX", T=0, C=seg_indices).astype(bool)

            # Create a 3-channel 2D depth map (Y, X, C)
            depth_maps_list = []
            for c in range(full_seg_masks_zcyx.shape[1]): # Iterate over C dimension
                mask_3d = full_seg_masks_zcyx[:, c, :, :] # Get (Z, Y, X) for one channel
                depth_map_2d = create_depth_map(mask_3d, z_scale)
                depth_maps_list.append(depth_map_2d)
            
            # Stack into a (Y, X, C) array
            depth_map_target = np.stack(depth_maps_list, axis=-1)

            # ----------------------------------------------------------
            # 4. Save to .npz file
            # ----------------------------------------------------------
            np.savez_compressed(
                save_path,
                full_stack=full_raw_stack,    # Shape: (Z, Y, X, 3)
                depth_map=depth_map_target,   # Shape: (Y, X, 3)
                z_step=np.float32(z_scale),   # Scalar
                total_slices=np.int32(total_slices) # Scalar
            )

        except Exception as e:
            print(f"  ERROR processing cell {cell_id}: {e}. Skipping this cell.")

    print("\n--- Preprocessing Finished ---")
    print(f"Successfully processed and saved data to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()