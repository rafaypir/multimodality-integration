import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# ------------------ USER CONFIG ------------------
PATIENT_ID_TO_VISUALIZE = "<PATIENT_ID>"
TARGET_SVS_DIR = "<PATH_TO_SVS_DIR>"
OUTPUT_DIR = "<PATH_TO_MODEL_OUTPUT_DIR>"
EPOCH_FOLDER_NAME = "<EPOCH_FOLDER_NAME>"
TARGET_PATHWAYS = [
    "GO:0006955", "GO:0001525", "GO:0022402", "GO:0007155",
    "GO:0006259", "GO:0030198", "GO:0030199", "GO:0007268",
    "GO:0006937", "GO:0030154", "GO:0009653", "GO:0008284",
    "GO:0006281", "GO:0030031"
]
PATHWAY_SCORES_PATH = "<PATH_TO_PATHWAY_SCORES_TSV>"
MASTER_SAVE_DIR = "<PATH_TO_ANALYSIS_OUTPUT>"
PATCH_SIZE = 256
PATCH_ALPHA = 0.7
BACKGROUND_LEVEL = 2
SELECTED_COLORMAP = 'coolwarm'
SELECTED_NORMALIZATION = 'Power'
GAMMA_P2W = 1
GAMMA_W2P = 1 
HIGHLIGHT_TOP_PATCHES = True
HIGHLIGHT_BORDER_COLOR = 'yellow'
HIGHLIGHT_BORDER_WIDTH = 2.0
TILES_BASE_DIR = "<PATH_TO_TILES_DIR>"
NUM_TOP_PATCHES_TO_SHOW = 5

# ------------------ SCRIPT LOGIC ------------------

def find_svs_file(patient_id, search_directory):
    for root, _, files in os.walk(search_directory):
        for file in files:
            if patient_id in file and file.lower().endswith(".svs"):
                return os.path.join(root, file)
    return None

def get_colormap(name):
    if name == 'blue_yellow_red':
        return mcolors.LinearSegmentedColormap.from_list("custom_byr", ["#0000FF", "#FFFF00", "#FF0000"])
    return plt.get_cmap(name)

def get_tiling_parameters(svs_path):
    try:
        slide = openslide.OpenSlide(svs_path)
        mag_str = slide.properties.get("openslide.objective-power")
        base_mag = float(mag_str) if mag_str else 40.0
        for level in range(slide.level_count):
            downsample = slide.level_downsamples[level]
            effective_mag = base_mag / downsample
            if 19.8 <= effective_mag <= 20.2:
                return {"slide": slide, "use_level": level, "downsample_needed": False, "level0_downsample_for_20x": downsample}
        if abs(base_mag - 40) < 5:
            return {"slide": slide, "use_level": 0, "downsample_needed": True, "level0_downsample_for_20x": 2.0}
        if abs(base_mag - 20) < 5:
            return {"slide": slide, "use_level": 0, "downsample_needed": False, "level0_downsample_for_20x": 1.0}
        raise RuntimeError(f"Could not determine strategy for slide with mag {base_mag}")
    except Exception as e:
        print(f"ERROR determining tiling parameters: {e}")
        return None

def generate_wsi_heatmaps(patient_id, svs_path, save_dir, cmap, attention_scores, patch_coords_20x, all_pathway_names, power_gamma):
    tiling_params = get_tiling_parameters(svs_path)
    if not tiling_params: return
    slide = tiling_params["slide"]
    os.makedirs(save_dir, exist_ok=True)
    bg_level = BACKGROUND_LEVEL if 0 <= BACKGROUND_LEVEL < slide.level_count else slide.level_count - 1
    bg_downsample = slide.level_downsamples[bg_level]
    bg_img = slide.read_region((0, 0), bg_level, slide.level_dimensions[bg_level]).convert("RGB")

    for pathway_name in TARGET_PATHWAYS:
        if pathway_name not in all_pathway_names: continue
        pathway_index = all_pathway_names.index(pathway_name)
        pathway_attention = attention_scores[pathway_index, :]
        top_indices_set = set()
        if HIGHLIGHT_TOP_PATCHES:
            top_indices = np.argsort(pathway_attention)[-NUM_TOP_PATCHES_TO_SHOW:]
            top_indices_set = set(top_indices)
        vmin, vmax = pathway_attention.min(), pathway_attention.max()
        norm = mcolors.PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax) if SELECTED_NORMALIZATION == 'Power' else mcolors.TwoSlopeNorm(vmin=vmin, vcenter=np.median(pathway_attention), vmax=vmax)

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(bg_img, aspect='equal')

        for coord_index in range(len(patch_coords_20x)):
            x_20x, y_20x = patch_coords_20x[coord_index]
            level0_downsample = tiling_params["level0_downsample_for_20x"]
            x_level0, y_level0 = int(x_20x * level0_downsample), int(y_20x * level0_downsample)
            patch_size_level0 = int(PATCH_SIZE * level0_downsample)
            scaled_x, scaled_y = x_level0 / bg_downsample, y_level0 / bg_downsample
            scaled_size = patch_size_level0 / bg_downsample
            edge_color, line_width, z_order = (HIGHLIGHT_BORDER_COLOR, HIGHLIGHT_BORDER_WIDTH, 20) if coord_index in top_indices_set else (None, 0, 10)
            rect = Rectangle((scaled_x, scaled_y), scaled_size, scaled_size,
                             facecolor=cmap(norm(pathway_attention[coord_index])),
                             alpha=PATCH_ALPHA, edgecolor=edge_color, linewidth=line_width, zorder=z_order)
            ax.add_patch(rect)

        ax.set_title(f"Patient: {patient_id}\nPathway: {pathway_name}", fontsize=16)
        ax.axis('off')
        plt.tight_layout(pad=0)
        heatmap_path = os.path.join(save_dir, f"highlighted_heatmap_{pathway_name.replace(':', '_')}.png")
        plt.savefig(heatmap_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

def find_patient_tile_folder(base_dir, patient_id):
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith(patient_id):
            return os.path.join(base_dir, folder_name)
    return None

def extract_top_patches(patient_id, svs_path, save_dir, attention_scores, coords, all_pathway_names):
    patient_tile_folder = find_patient_tile_folder(TILES_BASE_DIR, patient_id)
    if not patient_tile_folder: return
    os.makedirs(save_dir, exist_ok=True)
    svs_stem = os.path.splitext(os.path.basename(svs_path))[0]
    report_lines = [f"Top {NUM_TOP_PATCHES_TO_SHOW} Patches Report for Patient: {patient_id}", "="*60]

    for pathway_name in TARGET_PATHWAYS:
        if pathway_name not in all_pathway_names: continue
        pathway_index = all_pathway_names.index(pathway_name)
        pathway_attention = attention_scores[pathway_index, :]
        top_indices = np.argsort(pathway_attention)[-NUM_TOP_PATCHES_TO_SHOW:][::-1]
        report_lines.append(f"\nPathway: {pathway_name}")
        report_lines.append("-" * 40)
        fig, axes = plt.subplots(1, NUM_TOP_PATCHES_TO_SHOW, figsize=(20, 5))
        fig.suptitle(f"Top {NUM_TOP_PATCHES_TO_SHOW} Patches for Pathway: {pathway_name}\nPatient: {patient_id}", fontsize=16)

        for i, patch_index in enumerate(top_indices):
            ax = axes[i]
            score = pathway_attention[patch_index]
            x_grid = coords[patch_index][0] // PATCH_SIZE
            y_grid = coords[patch_index][1] // PATCH_SIZE
            patch_filename = f"{svs_stem}_{int(x_grid)}_{int(y_grid)}.png"
            patch_filepath = os.path.join(patient_tile_folder, patch_filename)
            report_lines.append(f"  Rank {i+1}: Score={score:.4f}, Coords=({int(coords[patch_index][0])}, {int(coords[patch_index][1])}), Grid=({int(x_grid)}, {int(y_grid)}), File={patch_filename}")
            if os.path.exists(patch_filepath):
                img = Image.open(patch_filepath)
                ax.imshow(img)
                title = f"Rank {i+1}\nScore: {score:.4f}\nGrid: ({int(x_grid)}, {int(y_grid)})"
            else:
                ax.text(0.5, 0.5, 'Patch Not Found', ha='center', va='center', fontsize=9, color='red')
                title = f"Rank {i+1}\nScore: {score:.4f}\n(File not found)"
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        save_path = os.path.join(save_dir, f"top_patches_{pathway_name.replace(':', '_')}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    report_filepath = os.path.join(save_dir, 'top_patches_report.txt')
    with open(report_filepath, 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    svs_file_path = find_svs_file(PATIENT_ID_TO_VISUALIZE, TARGET_SVS_DIR)
    if not svs_file_path: return
    epoch_dir = os.path.join(OUTPUT_DIR, EPOCH_FOLDER_NAME)
    coords_path = os.path.join(epoch_dir, f"{PATIENT_ID_TO_VISUALIZE}_coords.npy")
    if not os.path.exists(coords_path): return
    grid_coords = np.load(coords_path)
    if grid_coords.size == 0: return
    pixel_coords = grid_coords * PATCH_SIZE
    pathway_df = pd.read_csv(PATHWAY_SCORES_PATH, sep="\t", index_col=0)
    all_pathway_names = pathway_df.columns.tolist()
    active_cmap = get_colormap(SELECTED_COLORMAP)
    attention_types = ['p2w', 'w2p']
    for att_type in attention_types:
        attention_path = os.path.join(epoch_dir, f"{PATIENT_ID_TO_VISUALIZE}_attention_{att_type}.npy")
        if not os.path.exists(attention_path): continue
        attention_scores = np.load(attention_path)
        if att_type == 'w2p':
            attention_scores = attention_scores.T
        current_gamma = GAMMA_P2W if att_type == 'p2w' else GAMMA_W2P
        patient_master_dir = os.path.join(MASTER_SAVE_DIR, PATIENT_ID_TO_VISUALIZE, att_type)
        heatmap_save_dir = os.path.join(patient_master_dir, "heatmaps")
        top_patches_save_dir = os.path.join(patient_master_dir, "top_patches")
        generate_wsi_heatmaps(PATIENT_ID_TO_VISUALIZE, svs_file_path, heatmap_save_dir, active_cmap, attention_scores, pixel_coords, all_pathway_names, power_gamma=current_gamma)
        extract_top_patches(PATIENT_ID_TO_VISUALIZE, svs_file_path, top_patches_save_dir, attention_scores, pixel_coords, all_pathway_names)

if __name__ == '__main__':
    main()
