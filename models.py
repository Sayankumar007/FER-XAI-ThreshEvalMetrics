import os
import torch
from torchvision import models
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
      super(Identity,self).__init__()
    def forward(self,x):
      return x
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ***************** VGG19 **************************

vgg = models.vgg19(pretrained=True)
# vgg_model.avgpool = Identity()
vgg.classifier = nn.Sequential(
nn.Linear(25088, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

vgg = vgg.to(device)

      
      
# ***************** ResNet18 **************************
   
resnet = models.resnet18(pretrained=True)
# resnet_model.avgpool = Identity()
resnet.fc = nn.Sequential(
nn.Linear(512, 128),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(128,12))

resnet = resnet.to(device)



# ***************** GoogLeNet **************************

googlenet = models.googlenet(pretrained=True)
# googlenet_model.avgpool = Identity()
googlenet.fc = nn.Sequential(
nn.Linear(1024, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

googlenet = googlenet.to(device)



# ***************** EfficientNet **************************

efficientnet = models.efficientnet_v2_s(pretrained=True)
# efficientnet_model.avgpool = Identity()
efficientnet.fc = nn.Sequential(
nn.Linear(25088, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

efficientnet = efficientnet.to(device)



# ***************** MobileNet **************************

mobilenet = models.mobilenet_v3_small(pretrained=True)
# mobilenet_model.avgpool = Identity()
mobilenet.fc = nn.Sequential(
nn.Linear(25088, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

mobilenet = mobilenet.to(device)




# ***************** DenseNet **************************

densenet = models.densenet121(pretrained=True)
# densenet_model.avgpool = Identity()
densenet.fc = nn.Sequential(
nn.Linear(25088, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

densenet = densenet.to(device)