import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# Ground truth study area bounding box (EPSG:32648)
STUDY_AREA_BOUNDS: Tuple[float, float, float, float] = (704595.0, 1146765.0, 708315.0, 1150485.0)

# Ensure required output directories exist
for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# Band mapping definitions
# Standardized band names: blue, green, red, nir, swir1, swir2
LANDSAT5_BAND_MAP: Dict[str, str] = {
    "blue": "B1",
    "green": "B2",
    "red": "B3",
    "nir": "B4",
    "swir1": "B5",
    "swir2": "B7"
}

LANDSAT8_BAND_MAP: Dict[str, str] = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "nir": "B5",
    "swir1": "B6",
    "swir2": "B7"
}

DEFAULT_SPECTRAL_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
DEFAULT_INDICES = ["ndvi", "nbr", "ndwi"]

@dataclass
class Config:
    data_dir: str = DATA_DIR
    data_source: str = "anhgoc"  # "anhgoc" (original full scenes) or "cut" (pre-cut chips)
    results_dir: str = RESULTS_DIR
    models_dir: str = MODELS_DIR
    plots_dir: str = PLOTS_DIR
    predictions_dir: str = PREDICTIONS_DIR
    
    # Sensor and spatial params
    target_crs: str = "EPSG:32648"
    study_bounds: Tuple[float, float, float, float] = STUDY_AREA_BOUNDS
    image_size: int = 124
    spectral_bands: List[str] = field(default_factory=lambda: list(DEFAULT_SPECTRAL_BANDS))
    include_indices: bool = True
    indices: List[str] = field(default_factory=lambda: list(DEFAULT_INDICES))
    
    # Computed property: number of input channels to the model
    @property
    def in_channels(self) -> int:
        n = len(self.spectral_bands)
        if self.include_indices:
            n += len(self.indices)
        return n
    
    # Model architecture
    arch: str = "unet"  # "unet", "deeplabv3plus", "fpn"
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    num_classes: int = 1  # Binary segmentation with BCE / Dice
    
    # Training hyperparameters
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    threshold: float = 0.5
    
    # Evaluation & Validation
    val_years: List[str] = field(default_factory=lambda: ["2011", "2014"])
