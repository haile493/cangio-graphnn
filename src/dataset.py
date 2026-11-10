import os
import re
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.config import (
    LANDSAT5_BAND_MAP,
    LANDSAT8_BAND_MAP,
    DEFAULT_SPECTRAL_BANDS,
    DEFAULT_INDICES,
    STUDY_AREA_BOUNDS
)
from src.indices import compute_all_indices


def parse_landsat_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parses Landsat filename to extract sensor, pathrow, date, and band.
    Example: cut_LT05_L2SP_124053_20061226_20200831_02_T1_SR_B1.TIF
             LT05_L2SP_124053_20061226_20200831_02_T1_SR_B1.TIF
             LC08_L2SP_124053_20140215_20200911_02_T1_SR_B1.TIF
    """
    basename = os.path.basename(filename)
    match = re.match(
        r"(?:cut_)?(LT05|LC08)_(?:L2SP|L2SR|L1TP)_(\d{6})_(\d{8})_.*?_(SR_B\d+|ST_B\d+)\.TIF",
        basename,
        re.IGNORECASE
    )
    if not match:
        return None
    sensor, pathrow, date, band = match.groups()
    # Normalize band identifier (e.g. 'SR_B1' -> 'B1')
    band_id = band.split("_")[-1].upper()
    return {
        "sensor": sensor.upper(),
        "pathrow": pathrow,
        "date": date,
        "band": band_id,
        "scene_id": f"{sensor}_{date}"
    }


def discover_scenes(data_dir: str, source: str = "anhgoc") -> List[Dict]:
    """
    Scans data directory for scene folders, groups bands by scene date,
    and pairs each scene with the corresponding ground truth mask if available.
    
    Args:
        data_dir: Root dataset directory
        source: 'anhgoc' (original full scenes), 'cut' (pre-cut chips), or 'auto'
    """
    scenes_dict: Dict[str, Dict] = {}
    
    # Iterate over year folders (Cut_2006_2007, Cut_2008, etc.)
    for year_folder in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year_folder)
        if not os.path.isdir(year_path):
            continue
            
        # Locate ground truth mask in year_path
        gt_files = glob.glob(os.path.join(year_path, "*vunggaydo*.tif"))
        gt_path = gt_files[0] if gt_files else None
        
        # Determine target image folder based on source
        target_subfolder = None
        if source == "anhgoc":
            anhgoc_dirs = [d for d in glob.glob(os.path.join(year_path, "*Anhgoc*")) if os.path.isdir(d)]
            if anhgoc_dirs:
                target_subfolder = anhgoc_dirs[0]
            else:
                # Fallback to Cut if Anhgoc not yet present
                cut_dirs = [d for d in glob.glob(os.path.join(year_path, "Cut_*")) if os.path.isdir(d)]
                target_subfolder = cut_dirs[0] if cut_dirs else year_path
        elif source == "cut":
            cut_dirs = [d for d in glob.glob(os.path.join(year_path, "Cut_*")) if os.path.isdir(d)]
            target_subfolder = cut_dirs[0] if cut_dirs else year_path
        else: # auto
            anhgoc_dirs = [d for d in glob.glob(os.path.join(year_path, "*Anhgoc*")) if os.path.isdir(d)]
            if anhgoc_dirs and any(f.lower().endswith(".tif") for f in os.listdir(anhgoc_dirs[0])):
                target_subfolder = anhgoc_dirs[0]
            else:
                cut_dirs = [d for d in glob.glob(os.path.join(year_path, "Cut_*")) if os.path.isdir(d)]
                target_subfolder = cut_dirs[0] if cut_dirs else year_path
                
        tif_files = glob.glob(os.path.join(target_subfolder, "*.TIF")) + glob.glob(os.path.join(target_subfolder, "*.tif"))
        
        for tif in tif_files:
            info = parse_landsat_filename(tif)
            if not info:
                continue
            
            scene_key = f"{year_folder}_{info['scene_id']}"
            if scene_key not in scenes_dict:
                scenes_dict[scene_key] = {
                    "year_folder": year_folder,
                    "sensor": info["sensor"],
                    "date": info["date"],
                    "scene_id": info["scene_id"],
                    "bands": {},
                    "groundtruth_path": gt_path,
                    "folder": target_subfolder
                }
            scenes_dict[scene_key]["bands"][info["band"]] = tif

    # Filter out scenes that don't have minimum required bands
    valid_scenes = []
    for key, sc in scenes_dict.items():
        band_map = LANDSAT5_BAND_MAP if sc["sensor"] == "LT05" else LANDSAT8_BAND_MAP
        has_all_bands = all(b_id in sc["bands"] for b_id in band_map.values())
        if has_all_bands:
            valid_scenes.append(sc)
            
    return valid_scenes


class CanGioDataset(Dataset):
    """
    PyTorch Dataset for Can Gio multi-spectral satellite imagery.
    Loads calibrated surface reflectance, standardizes bands between Landsat-5 and Landsat-8,
    computes spectral indices, and returns paired disturbance masks.
    """
    def __init__(
        self,
        scenes: List[Dict],
        spectral_bands: Optional[List[str]] = None,
        include_indices: bool = True,
        indices: Optional[List[str]] = None,
        augment: bool = False,
        require_gt: bool = True,
        study_bounds: Optional[Tuple[float, float, float, float]] = STUDY_AREA_BOUNDS
    ):
        if require_gt:
            self.scenes = [s for s in scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]
        else:
            self.scenes = scenes
            
        self.spectral_bands = spectral_bands or list(DEFAULT_SPECTRAL_BANDS)
        self.include_indices = include_indices
        self.indices = indices or list(DEFAULT_INDICES)
        self.augment = augment
        self.study_bounds = study_bounds

    def __len__(self) -> int:
        return len(self.scenes)

    def _read_band(self, filepath: str) -> Tuple[np.ndarray, Optional[object], str]:
        with rasterio.open(filepath) as src:
            if self.study_bounds is not None and (src.width > 200 or src.height > 200):
                from rasterio.windows import from_bounds, transform as window_transform
                window = from_bounds(*self.study_bounds, transform=src.transform)
                arr = src.read(1, window=window).astype(np.float32)
                trans = window_transform(window, src.transform)
            else:
                arr = src.read(1).astype(np.float32)
                trans = src.transform
            crs = str(src.crs)

            # Landsat Collection 2 L2SP scale factor: DN * 0.0000275 - 0.2
            # Values > 60000 are nodata / saturation fill
            valid_mask = (arr > 0) & (arr < 60000)
            calibrated = np.zeros_like(arr)
            calibrated[valid_mask] = arr[valid_mask] * 0.0000275 - 0.2
            # Clip surface reflectance to physical range [0.0, 1.0]
            calibrated = np.clip(calibrated, 0.0, 1.0)
            return calibrated, trans, crs

    def load_scene_numpy(self, scene: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        sensor = scene["sensor"]
        band_map = LANDSAT5_BAND_MAP if sensor == "LT05" else LANDSAT8_BAND_MAP
        
        # Read standard spectral bands
        bands_data = {}
        last_trans = None
        last_crs = "EPSG:32648"
        for band_name in self.spectral_bands:
            b_id = band_map[band_name]
            fpath = scene["bands"][b_id]
            b_arr, b_trans, b_crs = self._read_band(fpath)
            bands_data[band_name] = b_arr
            last_trans = b_trans
            last_crs = b_crs

        # Build channels list
        channels = [bands_data[b] for b in self.spectral_bands]
        
        # Compute spectral indices
        if self.include_indices:
            idx_dict = compute_all_indices(bands_data)
            for idx_name in self.indices:
                if idx_name in idx_dict:
                    val = (idx_dict[idx_name] + 1.0) / 2.0
                    channels.append(np.clip(val, 0.0, 1.0))
        
        image = np.stack(channels, axis=0).astype(np.float32)
        
        gt_path = scene.get("groundtruth_path")
        if gt_path and os.path.exists(gt_path):
            with rasterio.open(gt_path) as src:
                mask = src.read(1).astype(np.float32)
                if mask.shape != image.shape[1:]:
                    mask = mask[:image.shape[1], :image.shape[2]]
                mask = (mask > 0.5).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)  # Shape (1, H, W)
                crs = str(src.crs)
                transform = src.transform
        else:
            mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.float32)
            crs = last_crs
            transform = last_trans

        meta = {
            "scene_id": scene["scene_id"],
            "year_folder": scene["year_folder"],
            "sensor": sensor,
            "date": scene["date"],
            "crs": crs,
            "transform": list(transform) if transform else None
        }
        return image, mask, meta

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        scene = self.scenes[idx]
        image, mask, meta = self.load_scene_numpy(scene)

        # Apply data augmentations (D4 symmetry rotations & flips)
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2)
                mask = np.flip(mask, axis=2)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(1, 2))

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask))
        return image_tensor, mask_tensor, meta


class BiTemporalCanGioDataset(Dataset):
    """
    Dataset for Option 2: Bi-temporal pairs (T1, T2) -> Mask(T2).
    Generates pairs of scenes to detect disturbance dynamics between pre and post acquisitions.
    """
    def __init__(
        self,
        scenes: List[Dict],
        pairing_strategy: str = "consecutive",  # "consecutive", "baseline", "all"
        spectral_bands: Optional[List[str]] = None,
        include_indices: bool = True,
        indices: Optional[List[str]] = None,
        augment: bool = False,
        study_bounds: Optional[Tuple[float, float, float, float]] = STUDY_AREA_BOUNDS
    ):
        # Sort scenes chronologically by date
        sorted_scenes = sorted(scenes, key=lambda s: str(s.get("date", "")))
        self.base_ds = CanGioDataset(
            scenes=sorted_scenes,
            spectral_bands=spectral_bands,
            include_indices=include_indices,
            indices=indices,
            augment=False,
            require_gt=False,
            study_bounds=study_bounds
        )
        self.augment = augment
        self.pairs: List[Tuple[int, int]] = []
        
        # Build pairs based on strategy
        n = len(sorted_scenes)
        if pairing_strategy == "consecutive":
            for i in range(1, n):
                if sorted_scenes[i].get("groundtruth_path"):
                    self.pairs.append((i - 1, i))
        elif pairing_strategy == "baseline":
            base_idx = 0
            for i in range(1, n):
                if sorted_scenes[i].get("groundtruth_path"):
                    self.pairs.append((base_idx, i))
        elif pairing_strategy == "all":
            for i in range(n):
                for j in range(i + 1, n):
                    if sorted_scenes[j].get("groundtruth_path"):
                        self.pairs.append((i, j))
        else:
            raise ValueError(f"Unknown pairing_strategy: {pairing_strategy}")

        if not self.pairs and n >= 2:
            # Fallback if no groundtruth marked on secondary
            self.pairs = [(i - 1, i) for i in range(1, n)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        idx1, idx2 = self.pairs[idx]
        sc1 = self.base_ds.scenes[idx1]
        sc2 = self.base_ds.scenes[idx2]

        img1, _, meta1 = self.base_ds.load_scene_numpy(sc1)
        img2, mask2, meta2 = self.base_ds.load_scene_numpy(sc2)

        # Apply synchronized spatial augmentations
        if self.augment:
            if np.random.rand() > 0.5:
                img1 = np.flip(img1, axis=2)
                img2 = np.flip(img2, axis=2)
                mask2 = np.flip(mask2, axis=2)
            if np.random.rand() > 0.5:
                img1 = np.flip(img1, axis=1)
                img2 = np.flip(img2, axis=1)
                mask2 = np.flip(mask2, axis=1)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                img1 = np.rot90(img1, k, axes=(1, 2))
                img2 = np.rot90(img2, k, axes=(1, 2))
                mask2 = np.rot90(mask2, k, axes=(1, 2))

        t1 = torch.from_numpy(np.ascontiguousarray(img1))
        t2 = torch.from_numpy(np.ascontiguousarray(img2))
        m2 = torch.from_numpy(np.ascontiguousarray(mask2))

        meta = {
            "t1_scene_id": meta1["scene_id"],
            "t1_date": meta1["date"],
            "t2_scene_id": meta2["scene_id"],
            "t2_date": meta2["date"],
            "crs": meta2["crs"],
            "transform": meta2["transform"]
        }
        return t1, t2, m2, meta


class SequentialCanGioDataset(Dataset):
    """
    Dataset for Option 3: Temporal sequence (T timestamps) -> Mask(T).
    Sliding window across chronological satellite passes.
    Returns:
        seq_tensor: Shape (T, C, H, W)
        target_mask: Shape (1, H, W)
        meta: Dict metadata
    """
    def __init__(
        self,
        scenes: List[Dict],
        seq_len: int = 3,
        spectral_bands: Optional[List[str]] = None,
        include_indices: bool = True,
        indices: Optional[List[str]] = None,
        augment: bool = False,
        study_bounds: Optional[Tuple[float, float, float, float]] = STUDY_AREA_BOUNDS
    ):
        sorted_scenes = sorted(scenes, key=lambda s: str(s.get("date", "")))
        self.base_ds = CanGioDataset(
            scenes=sorted_scenes,
            spectral_bands=spectral_bands,
            include_indices=include_indices,
            indices=indices,
            augment=False,
            require_gt=False,
            study_bounds=study_bounds
        )
        self.seq_len = seq_len
        self.augment = augment
        self.sequences: List[List[int]] = []

        n = len(sorted_scenes)
        for end_idx in range(seq_len - 1, n):
            target_scene = sorted_scenes[end_idx]
            if target_scene.get("groundtruth_path"):
                seq_indices = list(range(end_idx - seq_len + 1, end_idx + 1))
                self.sequences.append(seq_indices)

        if not self.sequences and n >= seq_len:
            self.sequences = [list(range(i - seq_len + 1, i + 1)) for i in range(seq_len - 1, n)]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        indices = self.sequences[idx]
        imgs = []
        target_mask = None
        target_meta = None

        for i, s_idx in enumerate(indices):
            sc = self.base_ds.scenes[s_idx]
            img, mask, meta = self.base_ds.load_scene_numpy(sc)
            imgs.append(img)
            if i == len(indices) - 1:
                target_mask = mask
                target_meta = meta

        imgs_seq = np.stack(imgs, axis=0)  # Shape (T, C, H, W)

        # Synchronized augmentation across all time slices
        if self.augment:
            if np.random.rand() > 0.5:
                imgs_seq = np.flip(imgs_seq, axis=3)
                target_mask = np.flip(target_mask, axis=2)
            if np.random.rand() > 0.5:
                imgs_seq = np.flip(imgs_seq, axis=2)
                target_mask = np.flip(target_mask, axis=1)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                imgs_seq = np.rot90(imgs_seq, k, axes=(2, 3))
                target_mask = np.rot90(target_mask, k, axes=(1, 2))

        seq_tensor = torch.from_numpy(np.ascontiguousarray(imgs_seq))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(target_mask))

        meta = {
            "dates": [self.base_ds.scenes[i]["date"] for i in indices],
            "target_scene_id": target_meta["scene_id"],
            "target_date": target_meta["date"],
            "crs": target_meta["crs"],
            "transform": target_meta["transform"]
        }
        return seq_tensor, mask_tensor, meta
