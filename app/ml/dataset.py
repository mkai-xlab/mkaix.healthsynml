import glob
import os
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import cv2
import torchvision.transforms as transforms

class KneeXRayDataset(Dataset):
    def __init__(self, data_dirs: Union[str, List[str]] = None, transform=None, root: str = None, split_dir: str = None, data_dir: Union[str, List[str]] = None):
        """
        Args:
            data_dirs (Union[str, List[str]], optional): A single path or a list of paths to directories 
                                                         containing class folders (e.g., '.../MedicalExpert-I').
            transform (callable, optional): Optional transform to be applied on a sample.
            root (str, optional): Root directory of local dataset.
            split_dir (str, optional): Split directory name ('train' uses MedicalExpert-I, others use MedicalExpert-II).
            data_dir (Union[str, List[str]], optional): Alias for data_dirs.
        """
        if data_dir is not None:
            data_dirs = data_dir

        if root is not None and split_dir is not None:
            # Map train/val splits to the local dataset's MedicalExpert-I and MedicalExpert-II folders
            sub_dir = "MedicalExpert-I" if split_dir == "train" else "MedicalExpert-II"
            data_dirs = [os.path.join(root, sub_dir)]

        if data_dirs is None:
            raise ValueError("Either 'data_dirs' or ('root' and 'split_dir') must be provided to KneeXRayDataset.")

        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        else:
            self.data_dirs = data_dirs

        for d in self.data_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"[WinError 3] The system cannot find the path specified: '{d}'")
            
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Assume categories are consistent across all directories and get them from the first one.
        self.categories = sorted([d for d in os.listdir(self.data_dirs[0]) if os.path.isdir(os.path.join(self.data_dirs[0], d))])
        
        self.load_datas()

    def load_datas(self):
        """Loads image paths and their corresponding labels from all provided directories."""
        print(f"Loading data from: {self.data_dirs}")
        for data_dir in self.data_dirs:
            for i, category in enumerate(self.categories):
                category_path = os.path.join(data_dir, category)
                if not os.path.isdir(category_path):
                    print(f"Warning: Category '{category}' not found in '{data_dir}'. Skipping.")
                    continue
                
                for file_name in os.listdir(category_path):
                    self.image_paths.append(os.path.join(category_path, file_name))
                    self.labels.append(i)
        print(f"Found {len(self.image_paths)} images in total.")

    def load_image_from_path(self, image_path):
        """Loads an image from a path and converts it to RGB."""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise IOError(f"Could not read image at {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.load_image_from_path(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_merged_dataset(data_dirs: List[str], transform=None) -> ConcatDataset:
    """
    A helper function to create a single dataset by concatenating multiple KneeXRayDataset instances.
    This is an alternative, often cleaner way to combine datasets.
    
    Args:
        data_dirs (List[str]): A list of paths to directories.
        transform (callable, optional): The transform to apply.
        
    Returns:
        ConcatDataset: A single dataset composed of all individual datasets.
    """
    datasets = [KneeXRayDataset(data_dir=d, transform=transform) for d in data_dirs]
    return ConcatDataset(datasets)

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
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=5,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        SquarePadOpenCV(),
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class KaggleKneeOsteoarthritisDataset(Dataset):
    """
    Dataset class for the Kaggle Knee Osteoarthritis dataset.
    Loads knee joint images and their corresponding Kellgren-Lawrence (KL) severity grades (0 to 4).
    
    Expected directory structure:
        root/
        └── split_dir/ (e.g., 'train', 'val', 'test')
            ├── 0/ (Normal)
            ├── 1/ (Doubtful)
            ├── 2/ (Mild)
            ├── 3/ (Moderate)
            └── 4/ (Severe)
    """

    def __init__(self, root: str, split_dir: str, transform=None):
        """
        Args:
            root (str): Path to the root directory of the Kaggle dataset.
            split_dir (str): Split directory name (typically 'train', 'val', or 'test').
            transform (callable, optional): PyTorch/torchvision transform pipeline.
        """
        self.root = root
        self.split_dir = split_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
            
        # Get sorted class directories (0, 1, 2, 3, 4) to ensure deterministic label mapping
        class_names = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()
        ])
        
        print(f"Loading '{split_dir}' split from: {split_path}")
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            label = int(class_name)
            
            # Filter and list only valid image files (avoid hidden files/directories)
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            image_files = [
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(valid_extensions)
            ]
            
            for file_name in image_files:
                self.image_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(label)
                
            print(f"  - Grade {class_name}: Found {len(image_files)} images.")
            
        print(f"Total images loaded for '{split_dir}': {len(self.image_paths)}")

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        """
        Loads an image from path using OpenCV and converts it from BGR to RGB.
        
        Args:
            image_path (str): File path of the image.
            
        Returns:
            np.ndarray: The image in RGB format.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise IOError(f"Could not read image at: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def __getitem__(self, idx: int):
        """
        Returns the image and its label at the specified index.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.load_image_from_path(image_path)
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset split.
        """
        return len(self.image_paths)

class MendeleyKneeXrayDataset(Dataset):
    """
    Dataset class for the Mendeley KneeXrayData dataset.
    Loads knee joint images and their corresponding Kellgren-Lawrence (KL) severity grades (0 to 4).
    
    Expected directory structure:
        root/ (e.g. ClsKLData/kneeKL224)
        └── split_dir/ (e.g., 'train', 'val', 'test', 'auto_test')
            ├── 0/ (Normal)
            ├── 1/ (Doubtful)
            ├── 2/ (Mild)
            ├── 3/ (Moderate)
            └── 4/ (Severe)
    """

    def __init__(self, root: str, split_dir: str, transform=None):
        """
        Args:
            root (str): Path to the root directory of the Mendeley dataset.
            split_dir (str): Split directory name (typically 'train', 'val', 'test', or 'auto_test').
            transform (callable, optional): PyTorch/torchvision transform pipeline.
        """
        self.root = root
        self.split_dir = split_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
            
        # Get sorted class directories (0, 1, 2, 3, 4) to ensure deterministic label mapping
        class_names = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()
        ])
        
        print(f"Loading '{split_dir}' split from: {split_path}")
        for class_name in class_names:
            class_dir = os.path.join(split_path, class_name)
            label = int(class_name)
            
            # Filter and list only valid image files (avoid hidden files/directories)
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            image_files = [
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(valid_extensions)
            ]
            
            for file_name in image_files:
                self.image_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(label)
                
            print(f"  - Grade {class_name}: Found {len(image_files)} images.")
            
        print(f"Total images loaded for '{split_dir}': {len(self.image_paths)}")

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        """
        Loads an image from path using OpenCV and converts it from BGR to RGB.
        
        Args:
            image_path (str): File path of the image.
            
        Returns:
            np.ndarray: The image in RGB format.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise IOError(f"Could not read image at: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def __getitem__(self, idx: int):
        """
        Returns the image and its label at the specified index.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.load_image_from_path(image_path)
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset split.
        """
        return len(self.image_paths)



if __name__ == '__main__':
    root_path = "C:\\Users\\vietn\\.cache\\kagglehub\\datasets\\shashwatwork\\knee-osteoarthritis-dataset-with-severity\\versions\\1"
    dataset = KaggleKneeOsteoarthritisDataset(root = root_path, split_dir = "train", transform = transforms.ToTensor())
    image, label = dataset.__getitem__(0)

    image_np = image.numpy()

    cv2_image = np.transpose(image_np, (1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cv2_image = (cv2_image * std + mean) * 255
    cv2_image = np.clip(cv2_image, 0, 255).astype(np.uint8)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Image", cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




