from typing import Type, Dict, Any
from torch.utils.data import Dataset
from pathlib import Path

# Import all available dataset classes
from app.ml.dataset import KneeXRayDataset, KaggleKneeOsteoarthritisDataset, MendeleyKneeXrayDataset

# --- Registry Configuration ---

# Define the project root relative to this file's location (app/ml/dataset_registry.py)
# This makes path construction robust and independent of the current working directory.
PROJECT_ROOT = Path(__file__).parent.parent.parent

# DATASET_REGISTRY now stores both the class and its default root path.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "local": {
        "class": KneeXRayDataset,
        "default_path": PROJECT_ROOT / "data" / "Knee X-ray Images"
    },
    "kaggle": {
        "class": KaggleKneeOsteoarthritisDataset,
        "default_path": Path(r"C:\Users\vietn\.cache\kagglehub\datasets\shashwatwork\knee-osteoarthritis-dataset-with-severity\versions\1")
    },
    "mendeley": {
        "class": MendeleyKneeXrayDataset,
        "default_path": PROJECT_ROOT / "data" / "KneeXrayData" / "ClsKLData"
    }
}

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Retrieves the dataset class and its default root path from the registry.

    Args:
        dataset_name (str): The unique identifier for the dataset.

    Returns:
        A dictionary containing the 'class' and 'default_path' for the dataset.

    Raises:
        ValueError: If the provided dataset_name is not found in the registry.
    """
    dataset_info = DATASET_REGISTRY.get(dataset_name)
    
    if not dataset_info:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
        
    print(f"Retrieved info for dataset '{dataset_name}':")
    print(f"  - Class: {dataset_info['class'].__name__}")
    print(f"  - Default Path: {dataset_info['default_path']}")
    
    return dataset_info

def get_dataset_class(dataset_name: str) -> Type[Dataset]:
    """
    Retrieves the dataset class from the registry.
    """
    return get_dataset_info(dataset_name)["class"]

def get_dataset_default_path(dataset_name: str) -> Path:
    """
    Retrieves the default path for a dataset.
    """
    return get_dataset_info(dataset_name)["default_path"]

