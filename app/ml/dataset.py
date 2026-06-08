import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import timm
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

class KneeXRayDataset(Dataset):

    # load image path & label
    def load_datas(self):
        #iter each category
        # print(self.categories)
        for i, category in enumerate(self.categories):
            category_path = os.path.join(self.root, category)

            #iter file in category
            for file_path in os.listdir(category_path):
                self.image_paths.append(os.path.join(category_path, file_path))
                self.labels.append(i)


    # load image from path -> numpy array (rgb)
    def load_image_from_path(self,imagePath):
        img_bgr = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    def __init__(self, root, train_dataset_dir, validate_dataset_dir, transform , train=True ):
        self.root = root
        self.transform  = transform

        # determine which dir will use (train , val)
        if train :
            self.root = os.path.join(root, train_dataset_dir)
        else:
            self.root = os.path.join(root,validate_dataset_dir)

        # get all category (0 -> 4)
        self.categories = os.listdir(self.root)
        self.image_paths = []
        self.labels = []

        # load data 
        self.load_datas()

        
    
    # len of dataset
    def __len__(self):
        return len(self.labels)
        
    # get item
    def __getitem__(self, idx):
        image = self.load_image_from_path(self.image_paths[idx])
        return self.transform(image), self.labels[idx]

class SquarePadOpenCV(object):
    def __call__(self, image):
        h, w = image.shape[:2]
        max_wh = max(h, w)
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        
        padded_image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded_image

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        SquarePadOpenCV(),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        SquarePadOpenCV(),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


