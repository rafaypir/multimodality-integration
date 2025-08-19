import openslide
import os
import cv2
import numpy as np
import pathlib
import time
from tqdm import tqdm

def build_svs_file_map(base_dir):
    print(f"Scanning '{base_dir}' to build a map of all .svs files... this might take a moment.")
    svs_map = {p.name: p for p in pathlib.Path(base_dir).rglob('*.svs')}
    print(f"Scan complete. Found {len(svs_map)} total .svs files.")
    return svs_map

def generate_full_tissue_mask(slide, svs_path, slide_output_dir, target_mag=20.0):
    base_mag = float(slide.properties.get("openslide.objective-power", 40.0))
    target_downsample = base_mag / target_mag
    available_downsamples = np.array(slide.level_downsamples)
    read_level = np.argmin(np.abs(available_downsamples - target_downsample))
    read_downsample = slide.level_downsamples[read_level]
    resize_factor = target_downsample / read_downsample

    level0_dims = slide.level_dimensions[0]
    target_dims = (int(level0_dims[0] / target_downsample), int(level0_dims[1] / target_downsample))
    read_dims = slide.level_dimensions[read_level]

    print(f"  Generating in-memory mask for {svs_path.name} at {target_dims}...")
    
    full_mask = np.zeros(target_dims[::-1], dtype=np.uint8)

    chunk_size = 4096 
    for y_read in range(0, read_dims[1], chunk_size):
        for x_read in range(0, read_dims[0], chunk_size):
            chunk_w = min(chunk_size, read_dims[0] - x_read)
            chunk_h = min(chunk_size, read_dims[1] - y_read)

            level0_x = int(x_read * read_downsample)
            level0_y = int(y_read * read_downsample)
            
            chunk_img = slide.read_region((level0_x, level0_y), read_level, (chunk_w, chunk_h))
            chunk_np = np.array(chunk_img)[:, :, :3]

            if resize_factor != 1.0:
                target_chunk_dims = (int(chunk_w / resize_factor), int(chunk_h / resize_factor))
                chunk_np = cv2.resize(chunk_np, target_chunk_dims, interpolation=cv2.INTER_AREA)

            
            gray = cv2.cvtColor(chunk_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 15, 122)
            kernel = np.ones((112, 112), np.uint8)
            dilated = cv2.dilate(edges, kernel)
            eroded = cv2.erode(dilated, kernel)
            
            
            target_x = int(x_read / resize_factor)
            target_y = int(y_read / resize_factor)
            full_mask[target_y : target_y + eroded.shape[0], target_x : target_x + eroded.shape[1]] = eroded

    with open(slide_output_dir / "magnification_info.txt", "w") as f:
        f.write(f"Original Base Magnification: {base_mag}x\n")
        f.write(f"Target Magnification for Tiling: {target_mag}x\n")
        f.write(f"Native Level Read From: {read_level} (at {base_mag/read_downsample:.1f}x)\n")
        f.write(f"Final Tile Dimensions: {target_dims}\n")
        f.write(f"Manual Resize Needed: {'Yes' if resize_factor != 1.0 else 'No'}\n")

    return full_mask, target_dims, read_level, resize_factor, base_mag


def process_single_wsi(svs_path, output_dir, tile_size=256, target_mag=20.0, tissue_cutoff=0.95):
    slide = None
    try:
        slide = openslide.OpenSlide(svs_path)
        slide_output_dir = output_dir / svs_path.stem
        slide_output_dir.mkdir(exist_ok=True)

        tissue_mask, target_dims, read_level, resize_factor, base_mag = generate_full_tissue_mask(slide, svs_path, slide_output_dir)

        tiles_x = target_dims[0] // tile_size
        tiles_y = target_dims[1] // tile_size
        read_tile_size = (int(tile_size * resize_factor), int(tile_size * resize_factor))
        tiles_created = 0
        
        thumbnail_mask = cv2.resize(tissue_mask, (0, 0), fx=0.05, fy=0.05)
        cv2.imwrite(str(slide_output_dir / 'diagnostic_edges_20x.jpg'), thumbnail_mask)

        with open(slide_output_dir / "coordinates.log", "w") as log_f:
            for y in range(tiles_y):
                for x in range(tiles_x):
                    mask_tile = tissue_mask[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
                    
                    tissue_percent = np.mean(mask_tile) / 255.0
                    
                    if tissue_percent >= tissue_cutoff:
                        level0_x = int(x * tile_size * (base_mag / target_mag))
                        level0_y = int(y * tile_size * (base_mag / target_mag))
                        
                        region_np = slide.read_region((level0_x, level0_y), read_level, read_tile_size)
                        tile_rgb = np.array(region_np)[:, :, :3]
                        
                        if resize_factor != 1.0:
                            tile_to_save = cv2.resize(tile_rgb, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
                        else:
                            tile_to_save = tile_rgb
                        
                        tile_bgr = cv2.cvtColor(tile_to_save, cv2.COLOR_RGB2BGR)
                        tile_path = slide_output_dir / f"{svs_path.stem}_{x}_{y}.png"
                        cv2.imwrite(str(tile_path), tile_bgr)
                        
                        log_f.write(f"{tile_path},{x},{y},{tile_size},{tile_size}\n")
                        tiles_created += 1

        return tiles_created
    finally:
        if slide:
            slide.close()


def main():
    # --- Configuration: UPDATE THESE PATHS ---
    base_directory = pathlib.Path("/path/to/your/svs_file_directory")
    target_list_file = '/path/to/your/list_of_slides.txt'
    output_directory = pathlib.Path("/path/to/your/output_directory_for_tiles")
    tissue_cutoff_percent = 0.95
    
    # --- Setup ---
    output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    svs_file_map = build_svs_file_map(base_directory)
    
    try:
        with open(target_list_file, 'r') as f:
            target_basenames = {line.strip() for line in f if line.strip()}
        print(f"Found {len(target_basenames)} unique target files in '{target_list_file}'.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The target file list '{target_list_file}' was not found.")
        return

    existing_folders = {d.name for d in output_directory.iterdir() if d.is_dir()}
    
    files_to_process = []
    skipped_count = 0
    for basename in sorted(list(target_basenames)):
        if basename in existing_folders:
            skipped_count += 1
        else:
            files_to_process.append(f"{basename}.svs")

    print(f"Skipping {skipped_count} files that have already been processed.")
    print(f"Found {len(files_to_process)} new files to process.")
    if not files_to_process:
        print("No new files to process. Exiting.")
        return

    # --- SEQUENTIAL PROCESSING ---
    processed_files = {}
    failed_files = []
    
    for svs_name in tqdm(files_to_process, desc="Tiling WSI"):
        if svs_name not in svs_file_map:
            print(f"\nERROR: Could not find {svs_name} in the source directory.")
            failed_files.append(svs_name)
            continue
        
        try:
            svs_path = svs_file_map[svs_name]
            num_tiles = process_single_wsi(svs_path, output_directory, tissue_cutoff=tissue_cutoff_percent)
            processed_files[svs_name] = num_tiles
        except Exception as e:
            print(f"\nFATAL ERROR during processing of {svs_name}. Reason: {e}")
            failed_files.append(svs_name)

    # --- Final Summary ---
    total_tiles = sum(processed_files.values())
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 50)
    print("                 Tiling Process Summary")
    print("=" * 50)
    print(f"Successfully processed: {len(processed_files)} files")
    print(f"Failed to process:      {len(failed_files)} files")
    print(f"Total new tiles created:  {total_tiles}")
    print(f"Total processing time:  {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 50)

    summary_path = output_directory / "processing_summary.txt"
    with open(summary_path, "a") as f:
        f.write(f"\n\n--- Summary for run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Successfully processed: {len(processed_files)} files\n")
        f.write(f"Failed files: {len(failed_files)}\n")
        f.write(f"Total new tiles: {total_tiles}\n")
        f.write(f"Processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write("\n--- Failed Files ---\n")
        for fname in sorted(failed_files):
            f.write(f"{fname}\n")
        f.write("\n--- Processed Files & Tile Counts ---\n")
        for fname, count in sorted(processed_files.items()):
            f.write(f"{fname}: {count} tiles\n")

    print(f"Detailed summary appended to: {summary_path.resolve()}")

if __name__ == "__main__":
    main()
