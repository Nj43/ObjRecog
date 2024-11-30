"""
Need https://github.com/google-research/augmix.git (see README.md)


This file contains the necessary data transformations.
"""

import torchvision.transforms as transforms
from AugMix import AugMixTransform


def ResNet152_data() :
    augmix_transform = AugMixTransform()
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), #fit the input size
        augmix_transform,  #augmix
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), #fit the input size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #normalization
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  #fit the input size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #normalization
    ])

    return train_transform, val_transform, test_transform



def VisionTransformer_data() :

    augmix_transform = AugMixTransform() 

    train_transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        augmix_transform, 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ]) 

    val_transform = transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ]) 

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  
    ])

    return train_transform, val_transform, test_transform



def basic_CNN_data() :

    data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return data_transforms, data_transforms, data_transforms


def DeiT_data():
    augmix_transform = AugMixTransform()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        augmix_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = val_transform 

    return train_transform, val_transform, test_transform


def ViT_EVA_Giant_data():
    augmix_transform = AugMixTransform(target_size=(336,336))

    train_transform = transforms.Compose([
        transforms.Resize((336,336)),
        augmix_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((336,336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    test_transform = val_transform

    return train_transform, val_transform, test_transform