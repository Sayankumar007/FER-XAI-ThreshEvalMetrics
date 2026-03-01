import torch
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as utils
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from operator import truediv
from cams import *
from train import InRAMImageFolder

# ==================== Traditional Metrics Functions ====================

def metrics(model, dataloader, cam_model_list=[]):
    """
    Compute metrics for a given model on a dataset.
    
    Loops through the dataset using a DataLoader and evaluates each CAM model 
    on the input data.
    
    Args:
        model (nn.Module): The CNN model to evaluate.
        dataloader (DataLoader): PyTorch DataLoader for evaluation dataset.
        cam_model_list (list): List of CAM model instances to compute metrics.
    """
    loop = tqdm(dataloader)
    count = 0

    for idx, data in enumerate(loop):
        input_ = data[0]
        input_ = input_.squeeze()
        input_ = input_.unsqueeze(0)
        # input_ = input_.type(torch.float32)
        if torch.cuda.is_available():
            input_ = input_.cuda()
        count += 1

        for cam_model in cam_model_list:
            cam_model.metrics(model, input_)

        if count == 100:
            break

    for cam_model in cam_model_list:
        cam_model.percentize(count)


def win_metrics(predicted_confidence_dict, avg_win, avg_drop, avg_increase, names_exclude=[], filename='Model_metrics_CAM.csv'):
    """
    Compute winning percentages and organize metrics into a CSV file.
    
    Args:
        predicted_confidence_dict (dict): Dictionary storing predicted confidences per CAM.
        avg_win (dict): Dictionary to accumulate win counts per CAM.
        avg_drop (dict): Dictionary with average drop metrics per CAM.
        avg_increase (dict): Dictionary with average increase metrics per CAM.
        names_exclude (list): List of CAM names to exclude from the calculations.
        filename (str): Filepath to save the resulting CSV.
    """
    keys = list(predicted_confidence_dict.keys())
    avg_win_keys = list(avg_win.keys())
    num_total = len(predicted_confidence_dict[keys[0]])
    names = ['gradcam','gradcampp','layercam','scorecam']
    max_conf = -1000
    max_conf_item = 'k'

    for item in avg_win:
        avg_win[item] = 0

    for idx in range(num_total):
        for index, item in enumerate(names):
            if item in names_exclude:
                continue
            if predicted_confidence_dict[item][idx] > max_conf:
                max_conf = predicted_confidence_dict[item][idx]
                max_conf_item = item

        avg_win[max_conf_item] += 1
        max_conf = -1000
        max_conf_item = ''

    for item in avg_win:
        avg_win[item] = avg_win[item] * 100 / num_total

    average_drop = []
    average_increase = []
    average_win = []
    names_ = []
    for item in names:
        if item in names_exclude:
            continue
        names_.append(item)
        average_drop.append(avg_drop[item])
        average_increase.append(avg_increase[item])
        average_win.append(avg_win[item])

    data = {
        'name': names_,
        'Average_drop%': average_drop,
        '%Increase_in_Confidence': average_increase,
        'Win%': average_win
    }
    df = pd.DataFrame(data)
    df.to_csv(filename)
    print(filename)
    print(df)


def CAM_eval(model, model_dict, root_dir, folder, name):
    """
    Evaluate a model using traditional CAM metrics.
    
    Computes precision, recall, F-measure for GradCAM, GradCAM++, LayerCAM, ScoreCAM.
    Saves results in CSV file per model and CAM type.
    
    Args:
        model (nn.Module): CNN model to evaluate.
        model_dict (dict): CAM model dictionary containing architecture and layer info.
        root_dir (str): Root dataset directory.
        folder (str): Folder to save metrics.
        name (str): Name of the model.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    avg_drop = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    avg_increase = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    avg_win = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    predicted_confidence = {'gradcam':[],'gradcampp':[],'layercam':[],'scorecam':[]}

    VALID_DIRECTORY = f'{root_dir}/val'
    valid_dataset = InRAMImageFolder(root=VALID_DIRECTORY, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    gradcam = GradCAM(model_dict)
    gradcampp = GradCAMpp(model_dict)
    layercam = LayerCAM(model_dict)
    scorecam = ScoreCAM(model_dict)
    model_list = [gradcam, gradcampp, layercam, scorecam]

    metrics(model, valid_loader, model_list)

    predicted_confidence['gradcam'] = gradcam.predicted_confidence_cam_list[:]
    avg_drop['gradcam'] = gradcam.avg_drop
    avg_increase['gradcam'] = gradcam.avg_increase
    del gradcam

    predicted_confidence['gradcampp'] = gradcampp.predicted_confidence_cam_list[:]
    avg_drop['gradcampp'] = gradcampp.avg_drop
    avg_increase['gradcampp'] = gradcampp.avg_increase
    del gradcampp

    predicted_confidence['layercam'] = layercam.predicted_confidence_cam_list[:]
    avg_drop['layercam'] = layercam.avg_drop
    avg_increase['layercam'] = layercam.avg_increase
    del layercam

    predicted_confidence['scorecam'] = scorecam.predicted_confidence_cam_list[:]
    avg_drop['scorecam'] = scorecam.avg_drop
    avg_increase['scorecam'] = scorecam.avg_increase
    del scorecam

    win_metrics(
        predicted_confidence_dict=predicted_confidence,
        avg_win=avg_win,
        avg_drop=avg_drop,
        avg_increase=avg_increase,
        names_exclude=[],
        filename=f'./{folder}/metrics_cam_{name}.csv'
    )

    print('Completed Successfully')


def log_output(folder, name, cam_name, threshold_value, mean_v, mini, maxi):
    """
    Log precision, recall, and F-measure for a specific CAM and threshold.
    
    Creates text file per model and threshold in PR_metric_results folder.
    """
    os.makedirs(f'./{folder}/PR_metric_results/', exist_ok=True)

    f = open(f'./{folder}/PR_metric_results/metrics_{name}_{threshold_value}.txt', 'a')
    f.write(f"Model : {name} ----- CAM : {cam_name} ----- Threshold : {threshold_value} (for both original and saliency map masking) \nThe Result is ->\n")
    sentence1 = f'\nMean_Precision: {mean_v[0]}\nMinimum Precision: {mini[0]}\nMaximum Precision: {maxi[0]}\n'
    f.write(sentence1)
    
    sentence2 = f'\nMean_Recall: {mean_v[1]}\nMinimum Recall: {mini[1]}\nMaximum Recall: {maxi[1]}\n'
    f.write(sentence2)
    
    sentence3 = f'\nMean_F_measure: {mean_v[2]}\nMinimum F_measure: {mini[2]}\nMaximum F_measure: {maxi[2]}\n'
    f.write(sentence3)
    
    f.write("\n\n\n")
    f.close()




# ==================== Proposed Thresholding-based Metrics Function ====================

def CAM_Thresh_Eval(model_dict, root_dir, folder, name, visualize=False, n=5):
    """
    Evaluate CAM models with thresholding metrics and optionally save visualizations.
    
    Performs precision-recall calculations over multiple thresholds for GradCAM, 
    GradCAM++, LayerCAM, and ScoreCAM.
    
    Args:
        model_dict (dict): CAM model dictionary containing architecture and layer info.
        root_dir (str): Root dataset directory.
        folder (str): Folder to save results.
        name (str): Model name.
        visualize (bool): Whether to save intermediate visualizations.
        n (int): Number of sample images to save for visualization.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

    to_grayscale = transforms.Grayscale(num_output_channels = 1)
    
    VALID_DIRECTORY = f'{root_dir}/val'
    valid_dataset = InRAMImageFolder(root = VALID_DIRECTORY, transform = transform)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle = True)
    
    # Create folders if they do not exist
    os.makedirs(f"./{folder}/PR_Curves/", exist_ok=True)
    if visualize:
        os.makedirs(f"./{folder}/visuals/", exist_ok=True)

    thresholds = [0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75]
    gradcam = GradCAM(model_dict)
    gradcampp = GradCAMpp(model_dict)
    layercam = LayerCAM(model_dict)
    scorecam = ScoreCAM(model_dict)
    cams = {'gradcam':gradcam, 'gradcampp':gradcampp, 'layercam': layercam, 'scorecam':scorecam}

    for threshold_value in tqdm(thresholds, desc = 'threshold : ', leave=False):
        plt.figure(figsize=(20,6))
        
        for cam_name, cam_model in tqdm(cams.items(), desc='CAMs : ', leave = False):
            count = 0
            precisions=[]
            recalls = []
            f_scores = []
            
            for idx, img in enumerate(valid_loader):
                data = img[0]
                gray_image = to_grayscale(data)
                data = data.to('cuda')
                s_map = cam_model(data).cpu()
                s_map = s_map[0, 0, :, :].numpy()
                gray_image = gray_image[0, 0, :, :].numpy()
                
                # Normalize the Saliency map Values to 0-255
                s_map_min = s_map.min()
                s_map_max = s_map.max()
                
                # If the image has a single unique value, avoid division by zero
                if s_map_min == s_map_max:
                    normalized_s_map = np.zeros_like(s_map, dtype = np.uint8)
                else:
                    normalized_s_map = 255 * (s_map - s_map_min) / (s_map_max - s_map_min)
                    normalized_s_map = normalized_s_map.astype(np.uint8)
                    
                normalized_s_map = normalized_s_map/255
         
                # multiply the normalized slaiency map with the original grayscale image
                multiplied_image = gray_image * normalized_s_map
                
                # normalize the multiplied image values to 0-255
                image_min = multiplied_image.min()
                image_max = multiplied_image.max()
                
                # If the image has a single unique value, avoid division by zero
                if image_min == image_max :
                    normalized_image = np.zeros_like(multiplied_image, dtype=np.uint8)
                else:
                    normalized_image = 255 * (multiplied_image - image_min) / (image_max - image_min)
                    normalized_image = normalized_image.astype(np.uint8)
                    
                multiplied_image_normalized = normalized_image/255
                
                # binarize the normalized version of multiplied image..
                binarized_image = (multiplied_image_normalized > threshold_value).astype(np.uint8)
                
                # binarize the normalized version of saliency map..
                binarized_s_map = (normalized_s_map > threshold_value).astype(np.uint8)
                
                # intersection using logical AND
                intersection = np.logical_and(binarized_image, binarized_s_map).astype(np.uint8)
                
                # count the number of pixels in the intersection region
                intersection_pixel_count = np.sum(intersection)
                
                recall = intersection_pixel_count / np.sum(binarized_image)
                precision = intersection_pixel_count / np.sum(binarized_s_map)
                
                precisions.append(precision)
                recalls.append(recall)
                
                beta =0.3
                f_measure = ((1+beta**2) * precision * recall) / ((beta**2) * precision + recall)
                
                f_scores.append(f_measure)
                
                # Visualization: save `n` samples
                if visualize and count < n:
                    utils.save_image(data, f'{folder}/visuals/{count}_{cam_name}_original.png', normalize=True)
                    utils.save_image(gray_image, f'{folder}/visuals/{count}_{cam_name}_gray.png', normalize=True)
                    utils.save_image(normalized_s_map, f'{folder}/visuals/{count}_{cam_name}_normalized_smap.png', normalize=True)
                    utils.save_image(multiplied_image_normalized, f'{folder}/visuals/{count}_{cam_name}_multiplied.png', normalize=True)
                    utils.save_image(binarized_image, f'{folder}/visuals/{count}_{cam_name}_binarized_multiplied.png', normalize=True)
                    utils.save_image(binarized_s_map, f'{folder}/visuals/{count}_{cam_name}_binarized_smap.png', normalize=True)
                    utils.save_image(intersection, f'{folder}/visuals/{count}_{cam_name}_intersection.png', normalize=True)
                
                count = count+1
                if count == 100:
                    break
            
            recalls = np.array(recalls)
            precisions = np.array(precisions)
            
            # sort the data by recall values for visualization purpose
            sorted_indices = np.argsort(recalls)
            sorted_precisions = precisions[sorted_indices]
            sorted_recalls = recalls[sorted_indices]
            
            plt.plot(sorted_recalls, sorted_precisions, label=cam_name)
            mean_precision = sum(precisions) / len(precisions)
            mean_recall = sum(recalls) / len(recalls)
            mean_f = sum(f_scores) / len(f_scores)
            
            min_precision = min(precisions)
            max_precision = max(precisions)
            
            min_recall = min(recalls)
            max_recall = max(recalls)
            
            min_f = min(f_scores)
            max_f = max(f_scores)
            
            log_output(folder, name, cam_name, threshold_value, [mean_precision, mean_recall, mean_f], [min_precision, min_recall, min_f], [max_precision, max_recall, max_f])
            
        plt.title(
            f'Precision-Recall Curve for {name} (threshold : {threshold_value})',
            fontsize=20
        )
        plt.xlabel('Recalls', fontsize=18)
        plt.ylabel('Precisions', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=18,loc='best')

        # save the final plot
        plt.savefig(
            f'./{folder}/PR_Curves/Precision_recall_curves_{name}_{threshold_value}.png',
            dpi=300,
            bbox_inches='tight'
        )
        # show the plot
        # plt.show()