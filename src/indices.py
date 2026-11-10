import numpy as np
import torch
from typing import Dict, Union

def calculate_ndvi(nir: Union[np.ndarray, torch.Tensor], red: Union[np.ndarray, torch.Tensor], eps: float = 1e-7) -> Union[np.ndarray, torch.Tensor]:
    """Normalized Difference Vegetation Index (NDVI = (NIR - Red) / (NIR + Red))"""
    numerator = nir - red
    denominator = nir + red + eps
    return numerator / denominator

def calculate_nbr(nir: Union[np.ndarray, torch.Tensor], swir2: Union[np.ndarray, torch.Tensor], eps: float = 1e-7) -> Union[np.ndarray, torch.Tensor]:
    """Normalized Burn Ratio / Disturbance Index (NBR = (NIR - SWIR2) / (NIR + SWIR2))"""
    numerator = nir - swir2
    denominator = nir + swir2 + eps
    return numerator / denominator

def calculate_ndwi(green: Union[np.ndarray, torch.Tensor], nir: Union[np.ndarray, torch.Tensor], eps: float = 1e-7) -> Union[np.ndarray, torch.Tensor]:
    """Normalized Difference Water Index (NDWI = (Green - NIR) / (Green + NIR))"""
    numerator = green - nir
    denominator = green + nir + eps
    return numerator / denominator

def compute_all_indices(bands_dict: Dict[str, Union[np.ndarray, torch.Tensor]], eps: float = 1e-7) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Computes standard remote sensing indices from a dictionary containing
    'green', 'red', 'nir', 'swir1', 'swir2'.
    """
    results = {}
    if "nir" in bands_dict and "red" in bands_dict:
        results["ndvi"] = calculate_ndvi(bands_dict["nir"], bands_dict["red"], eps=eps)
    if "nir" in bands_dict and "swir2" in bands_dict:
        results["nbr"] = calculate_nbr(bands_dict["nir"], bands_dict["swir2"], eps=eps)
    if "green" in bands_dict and "nir" in bands_dict:
        results["ndwi"] = calculate_ndwi(bands_dict["green"], bands_dict["nir"], eps=eps)
    return results
