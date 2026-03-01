import torch
import torch.nn.functional as F

import numpy as np
import cv2
import os 
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from cams import *

paths = []

# ==================== Utility Functions ====================

def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse the normalization applied to an image tensor.
    Args:
        x (Tensor): Normalized image tensor (B, C, H, W)
        mean (list): Mean values used for normalization
        std (list): Standard deviation values used for normalization
    Returns:
        Tensor: De-normalized image tensor
    """
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to create a result image.
    Args:
        img (Tensor): Input image tensor (1, 3, H, W)
        cam (Tensor): CAM tensor (1, 1, H', W')
    Returns:
        Tensor: Synthesized image with heatmap overlay (1, 3, H, W)
    """
    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1)).float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    result = heatmap + img.cpu()
    result = result.div(result.max())
    return result


# ==================== CAM Plotting Functions ====================

def plot_GradCAM(img_name, model, folder, name):
    """
    Generate and save GradCAM visualization for a single image.
    """
    wrapped_model = GradCAM(model)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_name).convert('RGB')
    tensor = transform(img).unsqueeze(0).to('cuda')
    cam = wrapped_model(tensor)
    img = reverse_normalize(tensor)
    heatmap = visualize(img, cam.cpu())
    save_image(heatmap, f"./{folder}/Plots/{img_name.split('/')[-2]}_{img_name.split('/')[-1].split('.jpg')[0]}_{name}_GradCAM.png")


def plot_GradCAMpp(img_name, model, folder, name):
    """
    Generate and save GradCAM++ visualization for a single image.
    """
    wrapped_model = GradCAMpp(model)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_name).convert('RGB')
    tensor = transform(img).unsqueeze(0).to('cuda')
    cam = wrapped_model(tensor)
    img = reverse_normalize(tensor)
    heatmap = visualize(img, cam.cpu())
    save_image(heatmap, f"./{folder}/Plots/{img_name.split('/')[-2]}_{img_name.split('/')[-1].split('.jpg')[0]}_{name}_GradCAMpp.png")


def plot_ScoreCAM(img_name, model, folder, name):
    """
    Generate and save ScoreCAM visualization for a single image.
    """
    wrapped_model = ScoreCAM(model)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_name).convert('RGB')
    tensor = transform(img).unsqueeze(0).to('cuda')
    cam = wrapped_model(tensor)
    img = reverse_normalize(tensor)
    heatmap = visualize(img, cam.cpu())
    save_image(heatmap, f"./{folder}/Plots/{img_name.split('/')[-2]}_{img_name.split('/')[-1].split('.jpg')[0]}_{name}_ScoreCAM.png")


def plot_LayerCAM(img_name, model, folder, name):
    """
    Generate and save LayerCAM visualization for a single image.
    """
    wrapped_model = LayerCAM(model)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_name).convert('RGB')
    tensor = transform(img).unsqueeze(0).to('cuda')
    cam = wrapped_model(tensor)
    img = reverse_normalize(tensor)
    heatmap = visualize(img, cam.cpu())
    save_image(heatmap, f"./{folder}/Plots/{img_name.split('/')[-2]}_{img_name.split('/')[-1].split('.jpg')[0]}_{name}_LayerCAM.png")


# ==================== Batch CAM Plotting ====================

def plot_CAMS(model_dict, root_dir, folder, name, n=5):
    """
    Generate and save CAM visualizations for multiple images from the test set.
    Loops over each class folder and randomly selects 'n' images per class.
    Calls individual CAM plotting functions for GradCAM, GradCAM++, ScoreCAM, and LayerCAM.
    """
    test_dir = f"{root_dir}/test"
    os.makedirs(f"./{folder}/Plots/", exist_ok=True)

    # Loop over each class folder
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                for _ in n:
                    random_img = random.choice(images)
                    img_path = os.path.join(class_path, random_img)
                    paths.append(img_path)

    # Generate CAM visualizations for selected images
    for path in paths:
        plot_GradCAM(path, model_dict, folder, name)
        plot_GradCAMpp(path, model_dict, folder, name)
        plot_ScoreCAM(path, model_dict, folder, name)
        plot_LayerCAM(path, model_dict, folder, name)