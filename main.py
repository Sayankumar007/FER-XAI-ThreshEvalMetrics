import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from models import *
from plots import *
from metrics import *
from train import build_model

SEED = 42    # experiments are done with seed 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# ********** Definine model dictionaries for CAMs *************

vgg_dict = dict(type='vgg19', arch=vgg, layer_name='features_35',input_size=(224, 224))
resnet_dict = dict(type='resnet18', arch=resnet, layer_name='layer4_1',input_size=(224, 224))
googlenet_dict = dict(type='googlenet', arch=googlenet, layer_name='inception5b',input_size=(224, 224))
efficientnet_dict = dict(type='efficientnetv2s', arch=efficientnet, layer_name='features',input_size=(224, 224))
mobilenet_dict = dict(type='mobilenet', arch=mobilenet, layer_name='features',input_size=(224, 224))
densenet_dict = dict(type='densenet', arch=densenet, layer_name='features',input_size=(224, 224))



# ================== Model Runner Functions ==================
"""
Each function performs the full pipeline for a specific CNN model:
1. Build and train the model
2. Plot CAMs with optional visualization of 'n' samples
3. Evaluate CAMs using traditional metrics
4. Evaluate CAMs using threshold-based metrics

These functions are compatible with run_model.py and accept:
- folder: folder name to save outputs
- root_dir: dataset root directory
- epochs: number of training epochs (default=10)
- lr: learning rate (default=0.001)
- n: number of CAM samples to visualize (default=5)
- visualize: whether to save intermediate visualization images (default=False)
"""

def func_vgg(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(vgg, root_dir, folder, name="vgg", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(vgg_dict, root_dir, folder, "vgg", n=n)
    CAM_eval(vgg, vgg_dict, root_dir, folder, "vgg")
    CAM_Thresh_Eval(vgg_dict, root_dir, folder, "vgg", visualize=visualize, n=n)


def func_resnet(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(resnet, root_dir, folder, name="resnet", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(resnet_dict, root_dir, folder, "resnet", n=n)
    CAM_eval(resnet, resnet_dict, root_dir, folder, "resnet")
    CAM_Thresh_Eval(resnet_dict, root_dir, folder, "resnet", visualize=visualize, n=n)


def func_googlenet(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(googlenet, root_dir, folder, name="googlenet", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(googlenet_dict, root_dir, folder, "googlenet", n=n)
    CAM_eval(googlenet, googlenet_dict, root_dir, folder, "googlenet")
    CAM_Thresh_Eval(googlenet_dict, root_dir, folder, "googlenet", visualize=visualize, n=n)


def func_efficientnet(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(efficientnet, root_dir, folder, name="efficientnet", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(efficientnet_dict, root_dir, folder, "efficientnet", n=n)
    CAM_eval(efficientnet, efficientnet_dict, root_dir, folder, "efficientnet")
    CAM_Thresh_Eval(efficientnet_dict, root_dir, folder, "efficientnet", visualize=visualize, n=n)


def func_mobilenet(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(mobilenet, root_dir, folder, name="mobilenet", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(mobilenet_dict, root_dir, folder, "mobilenet", n=n)
    CAM_eval(mobilenet, mobilenet_dict, root_dir, folder, "mobilenet")
    CAM_Thresh_Eval(mobilenet_dict, root_dir, folder, "mobilenet", visualize=visualize, n=n)


def func_densenet(folder, root_dir, epochs=10, lr=0.001, n=5, visualize=False):
    model = build_model(densenet, root_dir, folder, name="densenet", EPOCHS=epochs, LEARNING_RATE=lr)
    plot_CAMS(densenet_dict, root_dir, folder, "densenet", n=n)
    CAM_eval(densenet, densenet_dict, root_dir, folder, "densenet")
    CAM_Thresh_Eval(densenet_dict, root_dir, folder, "densenet", visualize=visualize, n=n)



  
