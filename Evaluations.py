# import necessary libraries
import numpy as np
from scipy.spatial import cKDTree

# import necessary libraries
import numpy as np
from scipy.spatial import cKDTree

# Convert a binary image into a set of 3D coordinates
def image_to_coordinates(image):
    return np.column_stack(np.where(image > 0))

# Compute the 95th percentile of the Hausdorff Distance between two sets of points
def calculate_hd95(set1, set2):
    tree1, tree2 = cKDTree(set1), cKDTree(set2)
    dist1 = tree1.query(set2, k=1, p=2)[0]
    dist2 = tree2.query(set1, k=1, p=2)[0]
    hd95_distance = np.percentile(np.concatenate([dist1, dist2]), 95)
    return hd95_distance


# Calculate the Hausdorff Coefficient for each class given the true and predicted segmentation maps
def hausdorff_coef_metric_per_classes(probabilities: np.ndarray,
                                      truth: np.ndarray,
                                      threshold: float = 0.5,
                                      classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).astype(float)
    assert(predictions.shape == truth.shape)

    max_distance = np.sqrt(probabilities.shape[2]**2 + probabilities.shape[3]**2)

    # Loop over each image and compute the HD95 for each class
    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(0.0)  # Both sets are empty, implies zero distance
            elif truth_.sum() == 0 or prediction.sum() == 0:
                scores[classes[class_]].append(max_distance)  # Only one set is empty, implies maximum distance
            else:
                # Convert images to sets of coordinates
                prediction = image_to_coordinates(prediction)
                truth_ = image_to_coordinates(truth_)
                scores[classes[class_]].append(calculate_hd95(prediction, truth_))
    return scores



def compute_scores_per_classes(model, dataloader, classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    hausdorff_scores_per_classes = {key: list() for key in classes}  # new
    iou_scores_per_classes = {key: list() for key in classes}
    sen_scores_per_classes = {key: list() for key in classes}
    spf_scores_per_classes = {key: list() for key in classes}

    # Loop over each batch in the dataloader
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            # Compute the Dice Coefficient and Hausdorff Distance for each class
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            hausdorff_scores = hausdorff_coef_metric_per_classes(logits, targets)  # new
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)
            sen_scores = sen_coef_metric_per_classes(logits, targets)
            spf_scores = spf_coef_metric_per_classes(logits, targets)

            # Add the scores to the list for each class
            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in hausdorff_scores.keys():  # new
                hausdorff_scores_per_classes[key].extend(hausdorff_scores[key])  # new
                
            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

            for key in sen_scores.keys():
                sen_scores_per_classes[key].extend(sen_scores[key])
                
            for key in spf_scores.keys():
                spf_scores_per_classes[key].extend(spf_scores[key])

    return dice_scores_per_classes, hausdorff_scores_per_classes,iou_scores_per_classes, sen_scores_per_classes, spf_scores_per_classes  # include hausdorff


import matplotlib.pyplot as plt
import numpy as np

# Function to plot average metrics per class
def plot_metrics(metrics_dict, classes, metric_name):
    # Compute mean metrics per class
    mean_metrics = [np.mean(metrics_dict[cls]) for cls in classes]
    
    # Create bar plot with different colors for each class
    bars = plt.bar(classes, mean_metrics, color=['blue', 'green', 'red'])
    plt.title(f'Average {metric_name} per Class')  
    plt.xlabel('Class')  
    plt.ylabel(metric_name)  
    print(mean_metrics) 
    
    # Loop through each bar and add the metric value on top of the bar
    for bar, metric in zip(bars, mean_metrics):
        yval = bar.get_height()
        # Place the text at the top of the bar
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(metric, 2), ha='center', va='bottom')

    plt.show() 


