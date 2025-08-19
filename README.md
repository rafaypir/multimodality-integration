# Multimodal Cross-Attention for Cancer Outcome Prediction

This repository contains the official implementation for the Master's thesis project on integrating histopathology and transcriptomics data to predict clinical outcomes in cancer patients from The Cancer Genome Atlas (TCGA). The core of this work is a cross-attention fusion mechanism designed to effectively learn from both data modalities.

## Abstract

Predicting patient survival and tumor subtype is a critical challenge in computational oncology. While Whole Slide Images (WSIs) offer morphological insights and RNA-sequencing provides a view into molecular activity, these modalities are often analyzed in isolation. This project introduces a multimodal framework that uses a cross-attention mechanism to fuse features from both WSIs and transcriptomics. By enabling each modality to attend to the most salient features of the other, the model creates a rich, integrated representation for improved prediction of survival and tumor subtypes.

***

## Repository Structure

This repository is organized into scripts for data preprocessing, model training, and interpretation.

### Data Preprocessing

- `WSI_patching.py`: A utility script to tile high-resolution Whole Slide Images (WSIs) into smaller, manageable patches (e.g., 256x256 pixels) for feature extraction.
- `CONCH_featureExtraction.py`: Extracts patch-level features from WSIs using the pretrained **CONCH** model.
- `UNI_featureExtraction.py`: Extracts patch-level features from WSIs using the pretrained **UNI** model.
- `wFDR_calculation.py`: Preprocesses RNA-seq data to compute patient-specific pathway activity scores using a weighted False Discovery Rate (wFDR) approach.

### Model Training & Evaluation

These scripts contain the main cross-attention model architecture and the training/evaluation pipelines for different feature combinations.

- `gex_wWSI_CoAttention.py`: Trains and evaluates the cross-attention model using **raw gene expression** (Gex) and **patch-level WSI** features.
- `wFDR_wWSI_CoAttention.py`: Trains and evaluates the model using the **wFDR pathway scores** and **patch-level WSI** features.
- `aWSI_IC_CoAttention.py`: Trains and evaluates the model using transcriptomic features derived from **Independent Component Analysis (ICA)** and an **averaged, slide-level WSI** representation.

### Model Interpretation

- `attention_maps.py`: Generates and visualizes the attention weights from a trained model. This is used to interpret the model's decisions by highlighting which image regions and genes were most influential for a given prediction.

***

## Workflow

The end-to-end workflow for this project can be summarized in the following steps:

1.  **Prepare WSI Data**: Use `WSI_patching.py` to create a dataset of image patches from the raw WSIs.
2.  **Extract WSI Features**: Run either `CONCH_featureExtraction.py` or `UNI_featureExtraction.py` on the patches to generate histopathology feature embeddings.
3.  **Prepare Transcriptomic Data**: Process the raw RNA-seq counts to generate one of the three feature types: normalized gene expression, ICA components, or wFDR pathway scores (using `wFDR_calculation.py`).
4.  **Train the Model**: Select one of the main scripts (e.g., `gex_wWSI_CoAttention.py`) to train the cross-attention framework on a specific combination of prepared features.
5.  **Evaluate and Interpret**: The script will output evaluation metrics (e.g., C-index, AUROC). Afterwards, run `attention_maps.py` on the saved model checkpoints to generate visualizations for biological validation.

***

## Citation

If you use this code in your research, please consider citing the following thesis:

```bibtex
@mastersthesis{abdulrafaypirzada2025,
  author       = {Abdul Rafay Pirzada},
  title        = {A COMPUTATIONAL FRAMEWORK FOR CANCER CHARACTERIZATION BY HISTOPATHOLOGY AND TRANSCRIPTOMICS DATA INTEGRATION},
  school       = {University of Luxembourg},
  year         = {2025}
}
