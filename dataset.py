import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # sort for alginment between masks and outputs
        self.images = sorted([f for f in os.listdir(image_dir) if not f.startswith('._')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('._')])
        
        # Geometric and color transforms for images
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(), # [Height, Width, 3], channel have to come first
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # averages of ImageNet for statistical baseline
        ])
        
        # Mask transforms (Must remain Nearest Neighbor to avoid changing class values)
        self.mask_transform = T.Compose([
             # standarize to 256x256 by using Nearest Neighbour Interpolation to keep 3 fixed classes / pixel values
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(self.convert_mask_to_tensor)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask
    
    def convert_mask_to_tensor(self, img):
        """make sure you preserve distinction between pixel values 1,2,3"""
        return torch.from_numpy(np.array(img)).long() - 1