# Deep-Learning-Final-Project
ECE-GY: 7123 Deep Learning final project by Rohith Mullapudi, Akasha Tigalappanavara, Atman Wagle

# UNETR for Brain Tumor Segmentation

This repository contains code for training a UNETR model for brain tumor segmentation using PyTorch and MONAI library.

## Introduction

The UNETR model is a variant of the UNet architecture integrated with Transformer blocks for medical image segmentation tasks. This implementation includes the necessary components for data loading, model training, validation, and visualization.

## Requirements

- Python 3.6+
- PyTorch (with cuda)
- MONAI (1.2.0 preferred)
- Matplotlib
- NumPy

Install the required libraries using `pip install -r requirements.txt`.

## Usage

### Data Preparation

Ensure that the dataset is organized according to the DecathlonDataset structure. Adjust the `base_dir` variable in the script to point to the dataset directory.

### Training

Run the script `base_model.py` to start training the base UNETR model. Run the script `optimized_model.py` to start training the optimized UNETR model. Training parameters such as batch size, learning rate, and number of epochs can be adjusted within the script.

### Evaluation

After training, the script automatically evaluates the model on the validation dataset and saves the best performing model based on mean dice score.

## File Descriptions

- `base_model.py`: Train the base UNETR model and visualize the outputs.
- `optimized_model.py`: Train the optimized UNETR model and visualize the outputs.
- `hpc_run.sh`: The shell script to run it on NYU HPC.
- `requirements.txt`: List of required Python libraries.

## Results

Training progress and validation metrics such as mean dice score for whole tumor (WT), tumor core (TC), and enhancing tumor (ET) are logged during training. Visualization of input images, ground truth labels, and model predictions are also provided.
