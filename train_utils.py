# train_utils.py

import DataProcessing
from DataProcessing import config
import DataLoaders
from DataLoaders import BratsDataset, get_dataloader
import Metrics
from Metrics import BCEDiceLoss
import Trainer
from Trainer import Trainer
import Evaluations
import VisualizaionResults
import Models

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import warnings


def main(model_name, batch_size,lr, num_epochs, pretrained=None):
    if model_name == 'ADRUwAMS':
        from Models.ADRUwAMS import UNet3D as Model
        
    model_instance = Model()


    # If a pretrained model is specified, load the pretrained weights
    if pretrained:
        pretrained_path = os.path.join('pretrained_weights', pretrained)
        model_instance.load_state_dict(torch.load(pretrained_path))
        model_instance.eval()

    # Other training setup
    trainer = Trainer(
        net=model_instance,
        dataset=BratsDataset,
        criterion=BCEDiceLoss(),
        lr=5e-8,
        accumulation_steps=4,
        batch_size=batch_size,
        fold=0,
        num_epochs=num_epochs,
        path_to_csv=config.path_to_csv,
    )

    
    val_dataloader = get_dataloader(BratsDataset, 'train_data.csv', phase='test', fold=0)
    len(dataloader)
    model.eval()
    
    
    dice_scores, hausdorff_scores, iou_scores, sen_scores, spf_scores = compute_scores_per_classes(model, val_dataloader, ['WT', 'TC', 'ET'])
    # Call the plot_metrics function to plot 
    plot_metrics(dice_scores,  ['WT', 'TC', 'ET'], 'Dice Score')
    plot_metrics(hausdorff_scores,  ['WT', 'TC', 'ET'], 'HD95')
    plot_metrics(sen_scores, ['WT', 'TC', 'ET'], 'Sensitivity')
    plot_metrics(spf_scores, ['WT', 'TC', 'ET'], 'Specificity')
    
    results = compute_results(model, val_dataloader, 0.33)
    unique_ids = set([item for sublist in results['Id'] for item in sublist])


    # Flatten the lists of lists to make them one-dimensional
    flattened_ids = [item for sublist in results['Id'] for item in sublist]
    flattened_images = [item for sublist in results['image'] for item in sublist]
    flattened_gt = [item for sublist in results['GT'] for item in sublist]
    flattened_prediction = [item for sublist in results['Prediction'] for item in sublist]


    # List of IDs of the images that shows a bigger tumor
    ids_to_show = ['BraTS20_Training_184','BraTS20_Training_362','BraTS20_Training_180','BraTS20_Training_133']

    # Loop over the IDs we want to display
    for id_ in ids_to_show:
        # Get the index of the ID in the flattened_ids list
        idx = flattened_ids.index(id_)

        # Get the corresponding image, ground truth, and prediction
        img = flattened_images[idx]
        gt = flattened_gt[idx]
        prediction = flattened_prediction[idx]

        # Select the slice of the image and masks we want to display
        slice_idx = 50  

        # Process the image and masks for displaying
        img_processed = image_preprocessing(img, slice_idx)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = mask_preprocessing(gt, slice_idx)
        pred_mask_WT, pred_mask_TC, pred_mask_ET = mask_preprocessing(prediction, slice_idx)

        # Plot the processed image, ground truth mask, and prediction mask
        plot_result(img_processed, gt_mask_WT, gt_mask_TC, gt_mask_ET, pred_mask_WT, pred_mask_TC, pred_mask_ET, id_)
