import glob
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
import cv2
import torchvision.transforms as transforms

# --- Custom Transforms ---

class SquarePadOpenCV(object):
    """Pads a rectangular image to a square."""
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

class OpenCVCLAHE(object):
    """
    Applies CLAHE using OpenCV. This transform is compatible with multiprocessing DataLoaders
    as it creates the CLAHE object on-the-fly in the __call__ method.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # Store parameters, but do not create the unpickleable cv2.CLAHE object here.
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        # Create the CLAHE object within the call. This is necessary for multiprocessing (num_workers > 0).
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        clahe_l_channel = clahe.apply(l_channel)
        
        merged_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
        final_rgb_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)
        
        return final_rgb_image

def get_transforms(img_size=224):
    """
    Returns a tuple of training and validation transform pipelines.
    """
    train_transform = transforms.Compose([
        SquarePadOpenCV(),
        OpenCVCLAHE(),
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.ColorJitter(
            brightness=0.03,
            contrast=0.03
        ),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        SquarePadOpenCV(),
        OpenCVCLAHE(),
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def remove_duplicate_images(image_paths: List[str], labels: List[int], exclude_hashes: set = None, categories: List[str] = None):
    """
    Computes MD5 hashes of all image files, removes duplicates, and prints statistics.
    
    Args:
        image_paths: List of absolute or relative image file paths.
        labels: List of integer labels corresponding to the image paths.
        exclude_hashes: A set of MD5 hashes to exclude (e.g. from the train split to prevent val leakage).
        categories: Optional list of category names mapping label indices to names.
        
    Returns:
        Tuple of (deduplicated_image_paths, deduplicated_labels, unique_hashes)
    """
    import hashlib
    from collections import Counter
    
    total_found = len(image_paths)
    unique_paths = []
    unique_labels = []
    unique_hashes = set()
    
    internal_dup_count = 0
    leakage_count = 0
    
    for path, label in zip(image_paths, labels):
        hash_md5 = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            h = hash_md5.hexdigest()
        except Exception as e:
            print(f"Warning: Could not read image {path} for duplicate checking: {e}")
            continue
            
        if exclude_hashes and h in exclude_hashes:
            leakage_count += 1
            continue
            
        if h in unique_hashes:
            internal_dup_count += 1
            continue
            
        unique_hashes.add(h)
        unique_paths.append(path)
        unique_labels.append(label)
        
    class_counts = Counter(unique_labels)
    
    print(f"\n--- Dataset Statistics & Deduplication ---")
    print(f"  - Total image files found: {total_found}")
    print(f"  - Internal duplicates removed: {internal_dup_count}")
    if exclude_hashes:
        print(f"  - Cross-split leak duplicates removed: {leakage_count}")
    print(f"  - Unique images kept: {len(unique_paths)}")
    print(f"  - Class distribution:")
    
    for label_idx in sorted(class_counts.keys()):
        count = class_counts[label_idx]
        class_name = categories[label_idx] if categories and label_idx < len(categories) else f"Class {label_idx}"
        print(f"    * {class_name}: {count} images")
    print("-------------------------------------------\n")
    
    return unique_paths, unique_labels, unique_hashes

# --- Dataset Classes ---
# ... (The rest of the file remains the same)
class KneeXRayDataset(Dataset):
    def __init__(self, data_dirs: Union[str, List[str]] = None, transform=None, root: str = None, split_dir: str = None, data_dir: Union[str, List[str]] = None, exclude_hashes: set = None):
        if data_dir is not None: data_dirs = data_dir
        if root is not None and split_dir is not None:
            sub_dir = "MedicalExpert-I" if split_dir == "train" else "MedicalExpert-II"
            data_dirs = [os.path.join(root, sub_dir)]
        if data_dirs is None: raise ValueError("No data directory provided.")
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        for d in self.data_dirs:
            if not os.path.isdir(d): raise FileNotFoundError(f"Path not found: '{d}'")
        self.transform = transform
        self.exclude_hashes = exclude_hashes
        self.image_paths, self.labels = [], []
        self.image_hashes = set()
        self.categories = sorted([d for d in os.listdir(self.data_dirs[0]) if os.path.isdir(os.path.join(self.data_dirs[0], d))])
        self.load_datas()

    def load_datas(self):
        print(f"Loading data from: {self.data_dirs}")
        raw_paths = []
        raw_labels = []
        for data_dir in self.data_dirs:
            for i, category in enumerate(self.categories):
                category_path = os.path.join(data_dir, category)
                if not os.path.isdir(category_path): continue
                for file_name in os.listdir(category_path):
                    raw_paths.append(os.path.join(category_path, file_name))
                    raw_labels.append(i)
        self.image_paths, self.labels, self.image_hashes = remove_duplicate_images(
            raw_paths, raw_labels, exclude_hashes=self.exclude_hashes, categories=self.categories
        )

    def load_image_from_path(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise IOError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        image = self.load_image_from_path(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

class KaggleKneeOsteoarthritisDataset(Dataset):
    def __init__(self, root: str, split_dir: str, transform=None, exclude_hashes: set = None):
        self.root = root
        self.transform = transform
        self.exclude_hashes = exclude_hashes
        raw_paths, raw_labels = [], []
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path): raise FileNotFoundError(f"Split directory not found: {split_path}")
        class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()])
        print(f"Loading '{split_dir}' split from: {split_path}")
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            label = int(class_name)
            valid_extensions = ('.png', '.jpg', '.jpeg')
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_extensions)]
            for file_name in image_files:
                raw_paths.append(os.path.join(class_dir, file_name))
                raw_labels.append(label)
        self.image_paths, self.labels, self.image_hashes = remove_duplicate_images(
            raw_paths, raw_labels, exclude_hashes=self.exclude_hashes, categories=class_names
        )

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise IOError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        image = self.load_image_from_path(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

    def __len__(self) -> int: return len(self.image_paths)

class MendeleyKneeXrayDataset(Dataset):
    def __init__(self, root: str, split_dir: str, transform=None, exclude_hashes: set = None):
        self.root = root
        self.transform = transform
        self.exclude_hashes = exclude_hashes
        raw_paths, raw_labels = [], []
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path): raise FileNotFoundError(f"Split directory not found: {split_path}")
        class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()])
        print(f"Loading '{split_dir}' split from: {split_path}")
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            label = int(class_name)
            valid_extensions = ('.png', '.jpg', '.jpeg')
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_extensions)]
            for file_name in image_files:
                raw_paths.append(os.path.join(class_dir, file_name))
                raw_labels.append(label)
        self.image_paths, self.labels, self.image_hashes = remove_duplicate_images(
            raw_paths, raw_labels, exclude_hashes=self.exclude_hashes, categories=class_names
        )

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise IOError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        image = self.load_image_from_path(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

    def __len__(self) -> int: return len(self.image_paths)
