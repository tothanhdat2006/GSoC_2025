import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NGADataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Data augmentation for training
        self.val_augmentation = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Data augmentation for validation and inference

        self.imgs_path = []
        for file in sorted(os.listdir(data_path)):
            if file.endswith('.jpg') and os.path.isfile(data_path + '/' + file):
                try:
                    _ = Image.open(data_path + '/' + file).load() # Check if image is corrupted
                    self.imgs_path.append(file)
                except Exception as e:
                    continue
        
    def __len__(self):
        return len(self.imgs_path)

    def get_filename(self, idx):
        return self.imgs_path[idx]
    
    def get_image(self, filename):
        return Image.open(self.data_path + '/' + filename)
    
    # Get item for validation and inference
    def get_item_val(self, idx):
        img = Image.open(self.data_path + '/' + self.imgs_path[idx])
        img = self.val_augmentation(img)
        return {
            'images': torch.as_tensor(img).float().contiguous() # PIL
        }

    # Get item for training
    def __getitem__(self, idx):
        img = Image.open(self.data_path + '/' + self.imgs_path[idx])
        img = self.train_augmentation(img)
        return {
            'images': torch.as_tensor(img).float().contiguous() # PIL
        }
    