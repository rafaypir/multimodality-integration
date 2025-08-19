import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from timm import create_model
from huggingface_hub import login, hf_hub_download
import logging

# ================================================
# Configuration: PLEASE UPDATE THESE VALUES
# ================================================
# 1. Your Hugging Face access token.
HF_TOKEN = "your_hugging_face_token_here"

# 2. The root folder containing your WSI patch folders.
PATCHES_DIR = '/path/to/your/patches_directory'

# 3. The folder where the output feature files will be saved.
OUTPUT_DIR = '/path/to/your/output_directory'

# 4. Name of the log file to be created in the script's directory.
LOG_FILE = 'process_log.txt'
# ================================================

# ==========================
# Logging Setup
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ==========================
# Authenticate with Hugging Face
# ==========================
try:
    login(token=HF_TOKEN)
    logging.info("Authenticated with Hugging Face.")
except Exception as e:
    logging.error(f"Failed to authenticate with Hugging Face: {e}")
    raise

# ==========================
# Load the UNI Model
# ==========================
try:
    local_dir = "assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
    model = create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logging.info(f"UNI model loaded successfully on {device}.")
except Exception as e:
    logging.error(f"Failed to load the UNI model: {e}")
    raise

# ==========================
# Setup Preprocessing Transforms
# ==========================
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
logging.info("Preprocessing transforms created successfully.")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Embedding Extraction Function
# ==========================
def extract_embeddings(model, image, device):
    """
    Extracts embeddings from the given image using the specified model.
    """
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze(0)

# ==========================
# Start Processing
# ==========================
logging.info("Embedding extraction process started.")

# Iterate over each WSI folder
for wsi_folder in os.listdir(PATCHES_DIR):
    wsi_path = os.path.join(PATCHES_DIR, wsi_folder)
    if os.path.isdir(wsi_path):
        logging.info(f"Processing WSI: {wsi_folder}")

        patches = [os.path.join(wsi_path, img) for img in os.listdir(wsi_path) if img.lower().endswith('.png')]

        if not patches:
            logging.warning(f"No PNG patches found in WSI folder: {wsi_folder}")
            continue

        embeddings_list = []

        for patch_path in patches:
            try:
                with Image.open(patch_path) as img:
                    image = img.convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                embedding = extract_embeddings(model, image, device).cpu().numpy()

                embeddings_list.append([os.path.basename(patch_path)] + embedding.tolist())

                logging.info(f"Processed patch: {patch_path}")

            except Exception as e:
                logging.error(f"Error processing patch {patch_path}: {e}")

        if embeddings_list:
            output_csv = os.path.join(OUTPUT_DIR, f"{wsi_folder}_embeddings.csv")
            
            columns = ["patch_name"] + [f"feature_{i}" for i in range(embedding.shape[0])]
            try:
                df = pd.DataFrame(embeddings_list, columns=columns)
                
                df.to_csv(
                    output_csv,
                    index=False,
                    float_format="%.6f"
                )
                
                logging.info(f"Saved embeddings for WSI {wsi_folder} to {output_csv}")

            except ValueError as ve:
                logging.error(f"Failed to save embeddings for WSI {wsi_folder}: {ve}")
        else:
            logging.warning(f"No embeddings to save for WSI {wsi_folder}.")

logging.info("Embedding extraction process completed.")