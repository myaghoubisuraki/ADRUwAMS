import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter("ignore", UserWarning)
from pathlib import Path
from tqdm import tqdm
##############################
# Global Configuration & Utilities
##############################

# Creating a class for global configuration parameters
class GlobalConfig:
    root_dir = 'BraTS2020'
    train_root_dir = 'BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = 'BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'train_data.csv'
    seed = 55  

# Function to seed everything for reproducibility
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Creating an instance of the global configuration
config = GlobalConfig()
seed_everything(config.seed)  


##############################
# Data Processing
##############################

# Loading survival information data and mapping data
survival_info_df = pd.read_csv('BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
name_mapping_df = pd.read_csv('BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')

# Renaming a column
name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 

# Merging survival information data and mapping data
df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

# Adding paths to the dataframe
paths = []
for index, row  in df.iterrows():    
    id_ = row['Brats20ID']
    phase = id_.split("_")[-2]
    
    if phase == 'Training':
        path = os.path.join(config.train_root_dir, id_)
    else:
        path = os.path.join(config.test_root_dir, id_)
    paths.append(path)
df['path'] = paths


############################################################

def calculate_tumor_size(patient_dir, patient_id):
    mask_file_path = patient_dir / f'BraTS20_Training_{patient_id}_seg.nii'
    
    if not mask_file_path.exists():
        print(f"File not found: {mask_file_path}")
        return 0

    # Load the mask using nibabel
    mask = nib.load(mask_file_path.as_posix()).get_fdata()
    
    # Compute the tumor size: count the number of non-zero elements (assuming tumor region is non-zero)
    tumor_size = np.count_nonzero(mask)
    
    return tumor_size


tumor_sizes = []
for path, id_ in tqdm(zip(df['path'], df['Brats20ID'])):
    patient_dir = Path(path)
    patient_id = id_.split('_')[-1]  
    tumor_size = calculate_tumor_size(patient_dir, patient_id)
    tumor_sizes.append(tumor_size)

df['Tumor_Size'] = tumor_sizes

############################

df1=df.copy()

def calculate_class_voxel_counts(patient_dir, patient_id):
    mask_file_path = patient_dir / f'BraTS20_Training_{patient_id}_seg.nii'
    
    # Check if the file exists
    if not mask_file_path.exists():
        print(f"File not found: {mask_file_path}")
        return {1: 0, 2: 0, 4: 0}  
        
    # Load the mask using nibabel
    mask = nib.load(mask_file_path.as_posix()).get_fdata()
    
    # Count the number of non-zero elements for each class
    class_counts = {1: np.count_nonzero(mask == 1),
                    2: np.count_nonzero(mask == 2),
                    4: np.count_nonzero(mask == 4)}  
    
    return class_counts

# Initialize lists to store counts
class_1_counts = []
class_2_counts = []
class_4_counts = []  

for path, id_ in tqdm(zip(df1['path'], df1['Brats20ID'])):
    patient_dir = Path(path)
    patient_id = id_.split('_')[-1]  
    voxel_counts = calculate_class_voxel_counts(patient_dir, patient_id)
    
    class_1_counts.append(voxel_counts[1])
    class_2_counts.append(voxel_counts[2])
    class_4_counts.append(voxel_counts[4])  

# Add the counts to the DataFrame
df1['NET'] = class_1_counts
df1['ED'] = class_2_counts
df1['ET'] = class_4_counts  

###############################
# Calculate quintiles for each class
def calculate_quintiles(data, column):
    percentiles = [data[column].quantile(i / 6) for i in range(1, 6)]
    bins = [0] + percentiles + [np.inf]
    labels = [str(i) for i in range(6)]
    return pd.cut(data[column], bins=bins, labels=labels)

df1['NET_Categories'] = calculate_quintiles(df1, 'NET')
df1['ED_Categories'] = calculate_quintiles(df1, 'ED')
df1['ET_Categories'] = calculate_quintiles(df1, 'ET')
#################################

train_data = df1.loc[df1['Tumor_Size'].notnull()].reset_index(drop=True)
train_data.count()
train_data.to_csv("train_data_.csv", index=False)

##################################

percentiles = [train_data['Tumor_Size'].quantile(i/5) for i in range(1, 5)]
print("Percentiles: ", percentiles)

# Define the bin edges
bins = [0] + percentiles + [np.inf]

# Define the bin labels 
labels = [str(i) for i in range(5)]

# Create the new column in df
train_data['Tumor_Size_Categories'] = pd.cut(train_data['Tumor_Size'], bins=bins, labels=labels)

###################################
from sklearn.model_selection import StratifiedKFold, train_test_split

total_data_points = len(train_data)  # Define the total number of data points

# Split the dataset into a combined training and validation subset (90%) and a test subset (10%)
train_val_subset, test_df = train_test_split(train_data, test_size=0.1, random_state=config.seed, stratify=train_data['Tumor_Size_Categories'])

# Reset the indices of train_val_subset to avoid KeyError
train_val_subset = train_val_subset.reset_index(drop=True)


# Apply 10-fold Stratified Cross-Validation to the combined training (%80) and validation subset (10%)
skf = StratifiedKFold(n_splits=10, random_state=config.seed, shuffle=True)

# Assigning folds for the combined training and validation subset 
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_subset, train_val_subset['Tumor_Size_Categories'])):
    train_val_subset.loc[val_idx, 'fold'] = fold

# Select one fold for validation and the rest for training, for example using fold 0 as the validation set
val_df = train_val_subset[train_val_subset['fold'] == 0]
train_df = train_val_subset[train_val_subset['fold'] != 0]

# Printing the sizes of each set
print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
train_val_subset.to_csv("train_data.csv", index=False)
