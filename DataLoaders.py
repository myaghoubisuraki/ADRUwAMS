import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import nibabel as nib
from scipy import ndimage

# Creating a class for the Brain Tumor Segmentation (BraTS) dataset
class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", augmentations=None):
        self.df = df  
        self.phase = phase
        self.augmentations = augmentations if augmentations else []
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
 
    def __len__(self):
        return len(self.df) * (1 + len(self.augmentations))

 
    def __getitem__(self, idx):
        num_augmentations = len(self.augmentations)
        original_idx = idx % len(self.df)
        augmentation_idx = idx // len(self.df)

        id_ = self.df.loc[original_idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

        
        mask_path = os.path.join(root_path, id_ + "_seg.nii")
        mask = self.load_img(mask_path)
        shift = self.calculate_shift_for_centering(mask)
        
        # Load and process images and masks
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)
            img = self.apply_shift(img, shift)
            img = self.center_crop(img)
            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        mask = self.apply_shift(mask, shift)
        mask = self.center_crop(mask)
        mask = self.preprocess_mask_labels(mask)

        # Apply augmentations if any are provided
        if augmentation_idx > 0 and augmentation_idx <= num_augmentations:
            augmentation_function = self.augmentations[augmentation_idx - 1]
            img, mask = augmentation_function(img, mask) 

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        return {
            "Id": id_,
            "image": img,
            "mask": mask,
        }

    def calculate_shift_for_centering(self, mask_data):
        center_of_mass = np.round(ndimage.measurements.center_of_mass(mask_data)).astype(int)
        desired_center = np.array(mask_data.shape) // 2
        shift = desired_center - center_of_mass
        return shift

    def apply_shift(self, image_data, shift):
        shifted_image = np.zeros_like(image_data)
        start_idx = np.maximum(-shift, 0)
        end_idx = np.minimum(np.array(image_data.shape) - shift, image_data.shape)
        target_start_idx = np.maximum(shift, 0)
        target_end_idx = target_start_idx + (end_idx - start_idx)
        shifted_image[
            target_start_idx[0]:target_end_idx[0],
            target_start_idx[1]:target_end_idx[1],
            target_start_idx[2]:target_end_idx[2]
        ] = image_data[
            start_idx[0]:end_idx[0],
            start_idx[1]:end_idx[1],
            start_idx[2]:end_idx[2]
        ]
        return shifted_image
    
    # function to load an image given its path
    def load_img(self, file_path):
            data = nib.load(file_path)
            data = np.asarray(data.dataobj)
            return data
 
    # function to normalize an image
    def normalize(self, data: np.ndarray):
            data_min = np.min(data)
            return (data - data_min) / (np.max(data) - data_min)

    def center_crop(self, data: np.ndarray):
    # Crop the data using the specified coordinates
        return data[56:184, 56:184, 13:141]
 
    # function to preprocess the mask labels
    def preprocess_mask_labels(self, mask: np.ndarray):
            mask_WT = mask.copy()
            mask_WT[mask_WT == 1] = 1 
            mask_WT[mask_WT == 2] = 1 
            mask_WT[mask_WT == 4] = 1
 
            mask_TC = mask.copy()
            mask_TC[mask_TC == 1] = 1
            mask_TC[mask_TC == 2] = 0
            mask_TC[mask_TC == 4] = 1
 
            mask_ET = mask.copy()
            mask_ET[mask_ET == 1] = 0
            mask_ET[mask_ET == 2] = 0
            mask_ET[mask_ET == 4] = 1
 
            mask = np.stack([mask_WT, mask_TC, mask_ET])
            mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))  
            return mask

        
def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
    augmentations=None,
):
    df = pd.read_csv(path_to_csv)
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)
 
    if phase == "train":
        df = train_df
    elif phase == "valid":
        df = val_df
        augmentations = False 
    elif phase == "test":
        df = test_df.reset_index(drop=True)
        augmentations = False 
 

    shuffle = phase == "train"
    dataset = dataset(df, phase,augmentations=augmentations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataloader
