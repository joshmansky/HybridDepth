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

NUM_CELLS_TO_PROCESS = 1000 
DATASET_ROOT = "/content/HybridDepth/AllenCell"

PROCESSED_DIR = os.path.join(DATASET_ROOT, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
#   Helper Functions 
# ==============================================================================

def get_channel_indices(name_dict, group_key, channel_names):
    channel_list = name_dict[group_key]
    indices = [channel_list.index(name) for name in channel_names]
    return indices

def create_depth_map(mask_3d, z_scale):
    if not mask_3d.any():
        return np.zeros(mask_3d.shape[1:], dtype=np.float32)
    
    depth_indices = np.argmax(mask_3d, axis=0)
    any_mask_in_column = mask_3d.any(axis=0)
    depth_indices = np.where(any_mask_in_column, depth_indices, 0)
    depth_microns = depth_indices.astype(np.float32) * z_scale
    return depth_microns

# ==============================================================================
#   Main Preprocessing Script
# ==============================================================================

def main():
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

    for index, row in tqdm(manifest_df.head(NUM_CELLS_TO_PROCESS).iterrows(), total=NUM_CELLS_TO_PROCESS):
        
        cell_id = row['CellId']
        save_path = os.path.join(PROCESSED_DIR, f"cell_{cell_id}.npz")
        
        if os.path.exists(save_path):
            continue
            
        try:
            # 1. Load Metadata
            name_dict = eval(row['name_dict'])
            scale_list = eval(row['scale_micron'])
            z_scale = scale_list[0] 
            
            # 2. Load 3-Channel Input Stack (X)
            raw_image_pkg_path = row['crop_raw']
            with pkg[raw_image_pkg_path].open() as f:
                img_reader = AICSImage(f)
                raw_indices = get_channel_indices(name_dict, 'crop_raw', ['dna', 'membrane', 'structure'])
                full_raw_stack_czyx = img_reader.get_image_data("ZCYX", T=0, C=raw_indices)
            
            # Permute to (Z, Y, X, C)
            full_raw_stack = np.transpose(full_raw_stack_czyx, (0, 2, 3, 1)) 
            total_slices = full_raw_stack.shape[0]

            # 3. Load Segmentation Masks (Y)
            seg_image_pkg_path = row['crop_seg']
            with pkg[seg_image_pkg_path].open() as f:
                seg_reader = AICSImage(f)
                seg_indices = get_channel_indices(name_dict, 'crop_seg', 
                                  ['dna_segmentation', 'membrane_segmentation', 'struct_segmentation'])
                # Shape: (Z, C, Y, X)
                full_seg_masks_zcyx = seg_reader.get_image_data("ZCYX", T=0, C=seg_indices).astype(bool)

            # --- Create Depth Maps ---
            depth_maps_list = []
            for c in range(full_seg_masks_zcyx.shape[1]): 
                mask_3d = full_seg_masks_zcyx[:, c, :, :] 
                depth_map_2d = create_depth_map(mask_3d, z_scale)
                depth_maps_list.append(depth_map_2d)
            depth_map_target = np.stack(depth_maps_list, axis=-1)

            # ==========================================================
            # ### NEW CODE START: Apply Mask to Input ###
            # ==========================================================
            
            # 1. Create a "Union Mask" (Z, Y, X)
            # If any channel (DNA, Mem, Struct) has a mask, keep the pixel.
            # We collapse the C dimension (axis 1) using Logical OR.
            union_mask_zyx = np.any(full_seg_masks_zcyx, axis=1)

            # 2. Expand mask to (Z, Y, X, 1) for broadcasting against (Z, Y, X, 3)
            union_mask_broadcast = union_mask_zyx[..., np.newaxis]

            # 3. Apply mask to Raw Stack
            # This zeroes out everything that is NOT part of the target cell.
            masked_raw_stack = full_raw_stack * union_mask_broadcast

            # ==========================================================
            # ### NEW CODE END ###
            # ==========================================================

            # 4. Save to .npz file
            np.savez_compressed(
                save_path,
                full_stack=masked_raw_stack,  # <--- Changed from full_raw_stack to masked_raw_stack
                depth_map=depth_map_target,   
                z_step=np.float32(z_scale),   
                total_slices=np.int32(total_slices)
            )

        except Exception as e:
            print(f"  ERROR processing cell {cell_id}: {e}. Skipping this cell.")

    print("\n--- Preprocessing Finished ---")
    print(f"Successfully processed and saved data to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()