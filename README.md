# ADRUwAMS: an Adaptive Dual Residual U-Net with Attention Gate and Multiscale Spatial Attention Mechanisms for Brain Tumor Segmentation
<img width="434" alt="presentation_results" src="https://github.com/myaghoubisuraki/ADRUwAMS/assets/137851429/65d5e4e9-18fa-45b0-8521-bf815d3620a5">


__Description__

This ADRUwAMS model is an enhancement of the classic U-Net architecture, extended to 3D for the volumetric segmentation of brain tumors and further integrated with dual residual blocks, attention gate, and multiscale spatial attention mechanisms. These enhancements enable the network to focus on specific spatial regions, effectively capturing intricate details in 3D medical images.

__Prerequisites__
The implementation requires Python 3.6 or later, Pytorch 1.4 or later, and the following libraries:
*	numpy
*	pandas
*	nibabel
*	scikit-learn
*	pytorch

# Dataset
The BraTS (Brain Tumor Segmentation) 2020 dataset is used for training. Due to an issue with the name of a segmentation file in BraTS 2020, we used a corrected version of the dataset called "BraTS_2020_Fixed_355", available on Kaggle. Ensure to provide the correct path to it in the Global Config Class.


# Model Architecture
<img width="974" alt="proposedmethod5" src="https://github.com/myaghoubisuraki/ADRUwAMS/assets/137851429/317057e3-31ff-4e10-8607-79465287812e">



# Usage Guidance
Running the Training Script on Your Computer:

__Script Execution:__

The script can be run directly from the command line. You can specify different parameters to customize the training process. 
__By default:__
  * Model: ADRUwAMS (change using --model)
  * Batch size: 4 (change using --batch-size)
  * Number of epochs: 1 (change using --epochs)
  * Learning rate: 5e-4 (change using --lr)
  * No pretrained weights are loaded by default (specify path using --pretrained)

__Run the script with default parameters:__

```
python main.py --model ADRUwAMS
```

__To use custom parameters, for example:__

```
python main.py --model ADRUwAMS --batch-size 4 --epochs 20 --lr 5e-4
```

__If you wish to use a pretrained model under pretrained_weights directory:__
```
python main.py --model ADRUwAMS --pretrained pretrained_model.pth
```

__Running the Training Script on Kaggle:__
If you wish to use the pretrained weights, it's recommended to run the script on Kaggle. The model was trained in an environment that provides the benefits of multi-GPU support, which facilitates efficient training.


# Citation
```
The paper is submitted to the Diagnostics Journal (https://www.mdpi.com/journal/diagnostics), and it is under review.
@{Yaghoubi Suraki, M.; Hernandez-Castillo, C. ADRUwAMS: an Adaptive Dual Residual U-Net with Attention Gate and Multiscale Spatial Attention Mechanisms for Brain Tumor Segmentation}
```

# Acknowledgments
We acknowledge the [organizers of the BraTS dataset](https://www.med.upenn.edu/cbica/brats2020/data.html) for providing the data for this project and the contributors of the BraTS_2020_Fixed_355 dataset on Kaggle for their valuable correction. The dataloader code is inspired by the [BraTS20_3dUnet_3dAutoEncoder](https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder/notebook) on Kaggle.
