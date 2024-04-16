
# Function to compute and store results including ID, image, ground truth and prediction
def compute_results(model, dataloader, treshold=0.33):
    # Check if CUDA is available and set device to CUDA, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize a dictionary to store results
    results = {"Id": [],"image": [], "GT": [],"Prediction": []}

    # Disable gradient calculation
    with torch.no_grad():
        # Enumerate over the dataloader
        for i, data in enumerate(dataloader):
            # Extract image and target data from dataloader
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            # Transfer data to the device
            imgs, targets = imgs.to(device), targets.to(device)
            # Make predictions
            logits = model(imgs)
            # Apply sigmoid function to the output logits
            probs = torch.sigmoid(logits)
            
            # Convert probabilities to binary predictions using the provided threshold
            predictions = (probs >= treshold).float()
            # Transfer predictions to the CPU
            predictions =  predictions.cpu()
            # Transfer targets to the CPU
            targets = targets.cpu()
            
            # Append results to the dictionary
            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)
            
            # Stop after the first 30 images
            if (i > 30):    
                return results
        return results



# Function to preprocess the image
def image_preprocessing(image, slice_idx):
    """
    Returns the FLAIR image as a mask for overlaying ground truth and predictions.
    """
    # Remove dimensions of size 1 from the tensor, convert it to numpy array, move axes and rotate
    image = image.squeeze().cpu().detach().numpy()
    image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
    flair_img = np.rot90(image[0, :, :, slice_idx])
    return flair_img

# Function to preprocess the mask
def mask_preprocessing(mask, slice_idx):
    """
    Preprocesses the mask tensor and selects a specific slice.
    """
    # Remove dimensions of size 1 from the tensor, convert it to numpy array, move axes and rotate
    mask = mask.squeeze().cpu().detach().numpy()
    mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
    
    # Slice the mask for different labels and rotate
    mask_WT = np.rot90(mask[0, :, :, slice_idx])
    mask_TC = np.rot90(mask[1, :, :, slice_idx])
    mask_ET = np.rot90(mask[2, :, :, slice_idx])
    return mask_WT, mask_TC, mask_ET



import matplotlib.pyplot as plt
import numpy.ma as ma

# Function to plot MRI image, ground truth mask and prediction mask
def plot_result(image, gt_mask_WT, gt_mask_TC, gt_mask_ET, pred_mask_WT, pred_mask_TC, pred_mask_ET, id_):
    # Add up the masks for the different classes to get the total mask for ground truth and prediction
    gt_mask = gt_mask_WT + gt_mask_TC + gt_mask_ET
    pred_mask = pred_mask_WT + pred_mask_TC + pred_mask_ET

    # Create a figure with 1 row and 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display the MRI image
    ax[0].imshow(image, cmap='gray')
    ax[0].title.set_text(f'MRI Image: {id_}')

    # Display the ground truth mask
    ax[1].imshow(gt_mask)
    ax[1].title.set_text('Ground Truth')

    # Display the prediction mask
    ax[2].imshow(pred_mask)
    ax[2].title.set_text('Prediction')

    # Save the figure
    plt.savefig(f'{id_}_plot.png')  
    plt.show()  # Display the figure


