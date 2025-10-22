import os, glob, re, random, zipfile, urllib.request
import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_name(fname):
    """
    Extracts a consistent prefix for grouping images belonging to the same stack.
    """
    name = os.path.splitext(os.path.basename(fname))[0]
    # Match patterns like:
    # mcf-z-stacks-03212011_p[a-z0-9]+_s[12]_w[1]
    m = re.match(r"(mcf-z-stacks-\d+_[a-z0-9]+_s[12]_w[1])", name)
    return m.group(1) if m else None

def load_stack_from_paths(paths):
    """Loads a stack from a list of file paths."""
    stack = np.stack([io.imread(p).astype(np.float32) / 255.0 for p in paths], axis=0)
    return stack

def compute_depth_and_aif(stack, optimal_focus_index=16, z_step_um=2.0):
    """
    Returns a 2D map of absolute depth in microns and the All-in-Focus image.
    stack: [z, H, W]
    """
    # Compute focus measure (variance of Laplacian)
    focus_measures = np.stack([filters.laplace(s)**2 for s in stack])
    best_focus_indices = np.argmax(focus_measures, axis=0) # [H, W]

    # Convert slice index → absolute depth in microns
    depth_um = (best_focus_indices - optimal_focus_index) * z_step_um
    
    # Create AIF image by sampling pixels from the best-focused slice
    h, w = stack.shape[1:]
    I, J = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    aif_image = stack[best_focus_indices, I, J]
    
    return depth_um.astype(np.float32), aif_image.astype(np.float32)

def main():
    root_dir = "/content/HybridDepth/dataloader/BBBC006" # Change this to your dataset path
    processed_dir = os.path.join(root_dir, "processed")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # ------------------------------
    # 1. Download/Extract (if needed)
    # ------------------------------
    base_url = "https://data.broadinstitute.org/bbbc/BBBC006/BBBC006_v1_images_z_{:02d}.zip"
    for i in range(32):
        zip_path = os.path.join(root_dir, f"BBBC006_v1_images_z_{i:02d}.zip")
        extract_dir = os.path.join(root_dir, f"BBBC006_v1_images_z_{i:02d}")
        
        # This check already handles your second request:
        # It only downloads if the *extracted directory* doesn't exist.
        if not os.path.exists(extract_dir):
            print(f"Downloading {os.path.basename(zip_path)} ...")
            urllib.request.urlretrieve(base_url.format(i), zip_path)
            
            print(f"Extracting {os.path.basename(zip_path)} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
                
            # --- ADDED THIS ---
            # Delete the zip file after successful extraction
            print(f"Deleting {os.path.basename(zip_path)} ...")
            os.remove(zip_path)
            # ------------------
        else:
            print(f"Skipping {os.path.basename(extract_dir)}, already exists.")

    print("✅ Dataset download/check complete.")
    
    # ------------------------------
    # 2. Build mapping of sites across z-levels
    # ------------------------------
    z_dirs = sorted(glob.glob(os.path.join(root_dir, "BBBC006_v1_images_z_*", "BBBC006_v1_images_z_*")))
    if not z_dirs:
        print(f"Error: No z-level directories found in {root_dir}.")
        print("Please make sure the dataset is downloaded and extracted correctly.")
        return

    site_dict = {}
    for z_idx, z_dir in enumerate(z_dirs):
        for f in glob.glob(os.path.join(z_dir, "*.tif")):
            norm = normalize_name(f)
            if norm:
                site_dict.setdefault(norm, {})[z_idx] = f

    complete_sites = {s: mapping for s, mapping in site_dict.items() if len(mapping) == 32}
    print(f"✅ Found {len(complete_sites)} complete image stacks with 32 slices.")

    # ------------------------------
    # 3. Process and save
    # ------------------------------
    print(f"Processing sites and saving to {processed_dir}...")
    for site_name, mapping in tqdm(complete_sites.items()):
        
        save_path = os.path.join(processed_dir, f"{site_name}.npz")
        
        # Skip if already processed
        if os.path.exists(save_path):
            continue
            
        try:
            # Load the full 32-slice stack
            paths = [mapping[z] for z in sorted(mapping.keys())]
            full_stack = load_stack_from_paths(paths) # [32, H, W]
            
            # Compute depth and AIF
            depth_map, aif_image = compute_depth_and_aif(full_stack)
            
            # Save the results
            np.savez_compressed(
                save_path,
                stack=full_stack,    # [32, H, W]
                depth=depth_map,     # [H, W]
                aif=aif_image        # [H, W]
            )
        except Exception as e:
            print(f"Failed to process {site_name}: {e}")

    print("✅ Pre-processing complete.")

if __name__ == "__main__":
    main()