"""
Need https://github.com/google-research/augmix.git (see README.md)

This file contains the AugMix for data augmentation.
"""

import random
from PIL import Image
from augmix.augmentations import * 
import numpy as np
from torchvision import transforms 

class AugMixTransform: 
    def __init__(self, num_ops=4, alpha=1.0, width=3, mixture_width=3, target_size=(224, 224)): 
        self.num_ops = num_ops
        self.alpha = alpha
        self.width = width
        self.mixture_width = mixture_width
        self.target_size = target_size

    def apply_random_transform(self, image):
        """Applies a random selection of augmentations to increase diversity."""
        augmentation_list = [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
        ]
        augment = random.choice(augmentation_list)
        return augment(image)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        # Resize the image to the target size 
        image = image.resize(self.target_size, Image.Resampling.LANCZOS) 

        ws = np.float32(np.random.dirichlet([self.alpha] * self.width)) 
        m = np.float32(np.random.beta(self.alpha, self.alpha)) 

        # Mix augmentations
        mixed = np.zeros_like(np.array(image).astype(np.float32)) 
        for i in range(self.width):
            image_aug = image.copy()
            image_aug = self.apply_random_transform(image_aug)
            depth = self.num_ops if self.num_ops > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = random.choice(augmentations_all)
                image_aug = op(image_aug, level=3)

            image_aug = image_aug.resize(self.target_size, Image.Resampling.LANCZOS)
            mixed += ws[i] * np.array(image_aug).astype(np.float32)

        # Blend original and augmented images
        mixed_image = (1 - m) * np.array(image).astype(np.float32) + m * mixed
        return Image.fromarray(np.uint8(mixed_image))
    