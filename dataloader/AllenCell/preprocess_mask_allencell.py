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
    """
    Creates a 2D depth map from a 3D binary mask.
    """
    if not mask_3d.any():
        return np.zeros(mask_3d.shape[1:], dtype=np.float32)
    
    # Argmax finds the index of the first True value along axis 0 (Z-axis)
    depth_indices = np.argmax(mask_3d, axis=0)
    
    # Mask out background (where no part of the cell exists in the column)
    any_mask_in_column = mask_3d.any(axis=0)
    depth_indices = np.where(any_mask_in_column, depth_indices, 0)
    
    # Convert to microns
    depth_microns = depth_indices.astype(np.float32) * z_scale
    return depth_microns

# ==============================================================================
#   Main Preprocessing Script
# ==============================================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    print("Accessing Quilt data package...")
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")

    print("Fetching and loading the metadata.csv file...")
    try:
        if not os.path.exists("metadata.csv"):
            manifest_path = pkg['metadata.csv'].fetch()
            print(f"Successfully fetched metadata file: {manifest_path}")
        else:
            print("metadata.csv already exists. Loading...")

        # Load only the rows we need to save RAM
        manifest_df = pd.read_csv('metadata.csv', nrows=NUM_CELLS_TO_PROCESS)
        print(f"Successfully loaded metadata for {len(manifest_df)} cells.")
        
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return

    print(f"--- Starting preprocessing for {NUM_CELLS_TO_PROCESS} cells ---")
    print(f"Saving processed files to: {PROCESSED_DIR}")

    for index, row in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0]):
        
        cell_id = row['CellId']
        save_path = os.path.join(PROCESSED_DIR, f"cell_{cell_id}.npz")
        
        if os.path.exists(save_path):
            continue
        
        # Temp filenames to avoid Quilt object errors
        temp_raw_filename = f"temp_raw_{cell_id}.tiff"
        temp_seg_filename = f"temp_seg_{cell_id}.tiff"

        try:
            # 1. Load Metadata
            name_dict = eval(row['name_dict'])
            scale_list = eval(row['scale_micron'])
            z_scale = scale_list[0] 
            
            # ----------------------------------------------------------
            # 2. Load 3-Channel Input Stack (X)
            # ----------------------------------------------------------
            # Fetch and Load Raw
            pkg[row['crop_raw']].fetch(temp_raw_filename)
            img_reader = AICSImage(temp_raw_filename)
            
            # Get indices for [dna, membrane, structure]
            raw_indices = get_channel_indices(name_dict, 'crop_raw', ['dna', 'membrane', 'structure'])
            full_raw_stack_czyx = img_reader.get_image_data("ZCYX", T=0, C=raw_indices)
            
            # Permute to (Z, Y, X, C) -> (Z, Y, X, 3)
            full_raw_stack = np.transpose(full_raw_stack_czyx, (0, 2, 3, 1)) 
            total_slices = full_raw_stack.shape[0]

            # ----------------------------------------------------------
            # 3. Load Segmentation Masks (Y)
            # ----------------------------------------------------------
            # Fetch and Load Seg
            pkg[row['crop_seg']].fetch(temp_seg_filename)
            seg_reader = AICSImage(temp_seg_filename)
            
            # Get indices for [dna, mem, struct]
            seg_indices = get_channel_indices(name_dict, 'crop_seg', 
                                  ['dna_segmentation', 'membrane_segmentation', 'struct_segmentation'])
            
            # Shape: (Z, C, Y, X) -> (Z, 3, Y, X)
            full_seg_masks_zcyx = seg_reader.get_image_data("ZCYX", T=0, C=seg_indices).astype(bool)

            # ----------------------------------------------------------
            # 4. Create Depth Maps
            # ----------------------------------------------------------
            depth_maps_list = []
            for c in range(full_seg_masks_zcyx.shape[1]): 
                mask_3d = full_seg_masks_zcyx[:, c, :, :] 
                depth_map_2d = create_depth_map(mask_3d, z_scale)
                depth_maps_list.append(depth_map_2d)
            depth_map_target = np.stack(depth_maps_list, axis=-1)

            # ==========================================================
            # ### NEW CODE: Strict Per-Channel Masking ###
            # ==========================================================
            
            # 1. Transpose segmentation masks to match raw stack shape
            # Current: (Z, C, Y, X) -> Target: (Z, Y, X, C)
            seg_masks_zyxc = np.transpose(full_seg_masks_zcyx, (0, 2, 3, 1))

            # 2. Apply Mask: Element-wise multiplication
            # This multiplies:
            #   Raw Channel 0 (DNA) * Mask Channel 0 (DNA)
            #   Raw Channel 1 (Mem) * Mask Channel 1 (Mem)
            #   Raw Channel 2 (Str) * Mask Channel 2 (Str)
            #
            # Broadcasting rules handle the dimensions automatically.
            masked_raw_stack = full_raw_stack * seg_masks_zyxc

            # ==========================================================

            # 5. Save to .npz file
            np.savez_compressed(
                save_path,
                full_stack=masked_raw_stack,
                depth_map=depth_map_target,   
                z_step=np.float32(z_scale),   
                total_slices=np.int32(total_slices)
            )

        except Exception as e:
            print(f"  ERROR processing cell {cell_id}: {e}. Skipping this cell.")
        
        finally:
            # CLEANUP
            if os.path.exists(temp_raw_filename):
                os.remove(temp_raw_filename)
            if os.path.exists(temp_seg_filename):
                os.remove(temp_seg_filename)

    print("\n--- Preprocessing Finished ---")
    print(f"Successfully processed and saved data to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()