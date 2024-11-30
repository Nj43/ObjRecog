"""
This file contains different models that I experiment.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm 


nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNet152(nn.Module) :
    def __init__(self) :
        super(ResNet152, self).__init__()
   
        #Load the pre-trained model
        self.model = models.resnet152(pretrained=True)

        #Replace the classifier to match the number of classes
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, nclasses)
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class predictions.
        """
        return self.model(x)
    


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        #Load the pre-trained model
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True) 
        #Replace the classifier to match the number of classes
        self.model.head = nn.Linear(self.model.head.in_features, nclasses) 

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class predictions.
        """
        return self.model(x)
    


class DeiT(nn.Module):
    def __init__(self):
        super(DeiT, self).__init__()
        #Load the pre-trained model
        self.deit = timm.create_model('deit3_large_patch16_224.fb_in22k_ft_in1k', pretrained=True)
        #Replace the classifier to match the number of classes
        self.deit.head = nn.Linear(self.deit.head.in_features, nclasses)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class predictions.
        """
        return self.deit(x)
    

class ViT_MAE(nn.Module):
    def __init__(self):
        super(ViT_MAE, self).__init__()
        #Load the pre-trained model
        self.vit = timm.create_model('vit_large_patch16_224.mae', pretrained=True)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class predictions.
        """
        return self.vit(x)
    


class ViT_EVA_Giant(nn.Module):
    def __init__(self):
        super(ViT_EVA_Giant, self).__init__()
        #Load the pre-trained model
        self.eva = timm.create_model('eva_giant_patch14_336.clip_ft_in1k', pretrained=True)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with class predictions.
        """
        return self.eva(x)