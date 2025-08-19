import os
import torch
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained
import pandas as pd
from tqdm import tqdm

# --- Configuration: UPDATE THESE PATHS ---
# 1. Path to the pretrained model checkpoint file.
checkpoint_path = '/path/to/your/CONCH/pytorch_model.bin'

# 2. Path to the main directory containing all WSI patch folders.
main_patch_folder = '/path/to/your/main_patch_folder/'

# 3. Path to the directory where embeddings will be saved.
embeddings_folder = '/path/to/your/output_embeddings_folder/'

# 4. Path to the text file containing the list of WSI folder names to process.
txt_file_path = '/path/to/your/slide_list.txt'
# --- End of Configuration ---

# Define model configuration and load
model_cfg = 'conch_ViT-B-16'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
model = model.to("cuda")
model.eval()

# Create the output directory if it doesn't exist
os.makedirs(embeddings_folder, exist_ok=True)

# Define the path to the log file
log_file_path = os.path.join(embeddings_folder, 'embed_logs.txt')

# Read the folder names (identifiers) from the specified text file
with open(txt_file_path, 'r') as file:
    selected_wsi_folders = {line.strip() for line in file}

# Function to log messages
def log_message(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    print(message)

# Start logging
log_message("Embedding extraction process started.")

# Initialize a dictionary to store paths and embeddings for all WSIs
all_wsi_embeddings = {}

# Loop through each WSI folder in the main patch directory
for wsi_folder in tqdm(os.listdir(main_patch_folder), desc="Processing WSIs"):
    wsi_path = os.path.join(main_patch_folder, wsi_folder)
    
    # Process only if the folder name is in our selected list
    if wsi_folder in selected_wsi_folders:
        log_message(f"Processing WSI: {wsi_folder}")
        
        # Check if embeddings CSV for this WSI already exists
        output_path = os.path.join(embeddings_folder, f'{wsi_folder}_embeddings.csv')
        if os.path.exists(output_path):
            log_message(f"Skipping {wsi_folder}: embeddings already exist.")
            continue

        # Initialize list to store embeddings for patches of the current WSI
        embeddings_list = []
        
        # Loop through each patch in the current WSI folder
        for patch_name in os.listdir(wsi_path):
            patch_path = os.path.join(wsi_path, patch_name)
            
            # Process only .png files
            if patch_path.endswith('.png'):
                image = Image.open(patch_path)
                image = preprocess(image).unsqueeze(0).to("cuda")
                
                with torch.inference_mode():
                    image_emb = model.encode_image(image)
                
                # Store the embedding and corresponding patch name
                embeddings_list.append({
                    'patch_name': patch_name,
                    'embedding': image_emb.squeeze().cpu().numpy()
                })
        
        # Convert embeddings for the current WSI to a DataFrame
        df = pd.DataFrame(embeddings_list)
        embeddings_df = pd.DataFrame(df['embedding'].tolist())
        
        # Combine patch names with embeddings into a single DataFrame
        result_df = pd.concat([df['patch_name'], embeddings_df], axis=1)
        
        # Save the DataFrame to CSV within the embeddings folder
        result_df.to_csv(output_path, index=False)
        
        # Store the embeddings in a dictionary (optional)
        all_wsi_embeddings[wsi_folder] = result_df
        log_message(f"Embeddings for WSI {wsi_folder} saved to {output_path}")
    else:
        # This part is optional but useful for debugging if some folders are unexpectedly skipped
        # log_message(f"Skipping WSI {wsi_folder}: not listed in {os.path.basename(txt_file_path)}")
        pass

log_message("All selected WSI embeddings have been processed and saved.")
