# Graph Neural Networks for Mangrove Canopy Disturbance Mapping (`cangio_graphnn`)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source PyTorch implementation of **Graph Neural Network (GNN) architectures and Bi-temporal Foundation Models** for high-resolution mangrove canopy disturbance mapping, blowdown corridor delineation, and ecological degradation monitoring in the UNESCO Can Gio Biosphere Reserve.

---

## 🌟 Key Features & Architectures

This repository implements, standardizes, and benchmarks **6 Graph Neural Network and Foundation architectures** adapted for multi-spectral Earth Observation (Landsat-5 TM and Landsat-8 OLI):

1. **BiG-SAM v2 (Graph-Augmented Foundation Model)**:
   - Fuses Segment Anything (SAM) foundation visual priors with remote sensing graph reasoning.
   - Combines Synergistic Parcel Characterization (SPC), Multi-Relational Knowledge Graphs (spatial, semantic, corridor veins), Dynamic Dilated KNN Max-Relative differencing, and Dense Multi-Scale Boundary Refinement.

2. **HRSEI-GNN** (*Jebur et al., J. Hazard. Mater. Adv. 2026*):
   - Biophysical Polarity Normalization separating positive and negative degradation indicators.
   - 4/8-nearest neighbor distance-weighted spatial grid message passing and ecological index estimation.

3. **Venagnosis PF-GNN** (*Hu & Wu, IEEE Access 2026*):
   - Multi-modal spectral and morphological curvature gradient extraction.
   - Multi-Relational Knowledge Graph Builder and Progressive Fusion Graph Network (PF-GNN) with GATv2 attention and hierarchical dynamic gating.

4. **GSRI-Net** (*Liu et al., ACM ICMR 2026*):
   - Graph-based Semantic Rerouting Interaction with Dynamic Dilated KNN Max-Relative Graph Convolutions.
   - Dual-PE (Sinusoidal + Relative) Supervised Cross-Attention for rotation-robust canopy corridor delineation.

5. **CGMN Multi-Branch** (*Shao et al., GIScience & Remote Sensing 2026*):
   - Tri-branch architecture: reciprocal spatial distance graph convolutions ($w_{ij} = 1 / d_{ij}$), pointwise nonlinear static MLP branch, and multi-scale/temporal GRU sequential branch.

6. **BiG-SAM v1**:
   - Baseline bipartite spatial-temporal graph segment anything network with prompt-guided token learner.

---

## 📁 Repository Structure

```
cangio_graphnn/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclass (default: 50 epochs)
│   ├── dataset.py                # Landsat calibration, spectral indices, bi-temporal pairing
│   ├── evaluate.py               # IoU, Dice/F1, Precision, Recall, Accuracy, Boundary IoU
│   ├── indices.py                # Normalized Difference Vegetation/Moisture Indices (NDVI, NBR, NDWI)
│   ├── models.py                 # Unified model builder & bi-temporal wrapper
│   └── models_fallback.py        # Standalone, dependency-free PyTorch implementations
├── scripts/
│   ├── prepare_dataset.py        # Dataset discovery, calibration check, and pair verification
│   ├── run_graph_benchmark.py    # Benchmark for single-date graph models (50 epochs)
│   ├── run_all_models_bitemporal_benchmark.py  # Comprehensive bi-temporal benchmark (50 epochs)
│   ├── run_big_sam_v2_benchmark.py             # BiG-SAM v2 training, ablation, and evaluation (50 epochs)
│   ├── run_big_sam_benchmark.py                # BiG-SAM v1 benchmark (50 epochs)
│   └── run_improved_singledate_benchmark.py    # Single-date improved benchmark (50 epochs)
├── train.py                      # Unified CLI training entry point (epochs=50 default)
├── evaluate.py                   # Unified CLI evaluation & threshold sweep entry point
├── requirements.txt              # Python package requirements
├── setup.py                      # Package installation script
├── pyproject.toml                # Modern build system configuration
├── LICENSE                       # MIT License
└── README.md
```

> **Note**: This repository contains exclusively source code for data preparation, model training, evaluation, and benchmarking. All manuscript writing, LaTeX/Word generation, and figure plotting code have been strictly omitted.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/cangio-earth/cangio_graphnn.git
cd cangio_graphnn
```

### 2. Create and activate a Python environment
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🛰️ Dataset Preparation

The pipeline expects calibrated surface reflectance GeoTIFF scenes organized by acquisition year in `data/`:

```
data/
├── Cut_2006_2007/       # Landsat-5 TM scenes + ground truth mask
├── Cut_2008/            # Landsat-5 TM scenes + ground truth mask
├── Cut_2010/            # Landsat-5 TM scenes + ground truth mask
├── Cut_2011/            # Landsat-5 TM scenes (Validation split)
└── Cut_2014/            # Landsat-8 OLI scenes (Validation split)
```

Each scene includes multi-spectral bands (`B1` through `B7`) and corresponding binary disturbance ground-truth rasters (`*vunggaydo*.tif`).

### Radiometric Calibration & Spatial Alignment:
- **Physical Scale Factor**: Landsat Collection 2 Level-2 Surface Reflectance ($DN \times 0.0000275 - 0.2$), clipped to $[0.0, 1.0]$.
- **Target Coordinate System**: WGS 84 / UTM Zone 48N (`EPSG:32648`).
- **Spectral Bands**: Blue, Green, Red, Near-Infrared (NIR), Shortwave Infrared 1 (SWIR1), Shortwave Infrared 2 (SWIR2).
- **Spectral Indices**: NDVI (Normalized Difference Vegetation Index), NBR (Normalized Burn Ratio), NDWI (Normalized Difference Water Index).

### Verify Dataset Integrity:
Run the preparation script to verify scene discovery, band calibration, and leak-free split:
```bash
python scripts/prepare_dataset.py --data-dir data --source anhgoc
```

---

## 🏋️ Model Training

All models default to **50 epochs** with the AdamW optimizer and Cosine Annealing learning rate schedule.

### 1. Train Bi-Temporal BiG-SAM v2 (Recommended)
```bash
python train.py --model big_sam_v2 --mode bitemporal --epochs 50 --lr 3e-4
```

### 2. Train Single-Date Graph Architectures
```bash
# Train HRSEI-GNN
python train.py --model hrsei_gnn --mode single --epochs 50 --lr 1.5e-3

# Train Venagnosis PF-GNN
python train.py --model venagnosis_pfgnn --mode single --epochs 50 --lr 1.5e-3

# Train GSRI-Net
python train.py --model gsri_net --mode single --epochs 50 --lr 1.5e-3

# Train CGMN Multi-Branch
python train.py --model cgmn_net --mode single --epochs 50 --lr 1.5e-3
```

### 3. Run Full Benchmark Suites
```bash
# Bi-temporal benchmark comparing all graph models (50 epochs)
python scripts/run_all_models_bitemporal_benchmark.py

# Single-date benchmark comparing 2026 graph models (50 epochs)
python scripts/run_graph_benchmark.py --epochs 50

# BiG-SAM v2 ablation benchmark (50 epochs)
python scripts/run_big_sam_v2_benchmark.py
```

---

## 📊 Model Evaluation

Evaluate any trained checkpoint or initialized model across validation scenes (`Cut_2011` and `Cut_2014`):

```bash
python evaluate.py \
    --model big_sam_v2 \
    --mode bitemporal \
    --checkpoint results/models/best_big_sam_v2_bitemp.pth
```

### Evaluated Metrics:
- **Dice Coefficient / F1 Score**: Harmonic mean of Precision and Recall.
- **Intersection over Union (IoU)**: Jaccard index measuring canopy disturbance overlap.
- **Precision & Recall**: Quantifying false alarms and missed canopy gaps.
- **Boundary IoU**: Boundary delineation sharpness metric (*Cheng et al., CVPR 2021*).
- **Optimal Decision Threshold Sweep**: Automatic threshold search across $[0.15, 0.85]$.

---

## 📜 Benchmark Results

Representative benchmark performance on the zero-leakage Can Gio validation set (Landsat-5 & Landsat-8):

| Model Architecture | Input Mode | Parameters | F1 Score | IoU | Boundary IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BiG-SAM v2 (Foundation Prior)** | Bi-Temporal | 12.8M | **0.8654** | **0.7628** | **0.7812** |
| **CGMN Multi-Branch** | Native 5D | 312K | 0.8124 | 0.6841 | 0.6953 |
| **Venagnosis PF-GNN** | Difference | 475K | 0.8031 | 0.6710 | 0.6840 |
| **GSRI-Net** | Difference | 542K | 0.7985 | 0.6646 | 0.6721 |
| **HRSEI-GNN** | Difference | 215K | 0.7842 | 0.6450 | 0.6519 |
| **BiG-SAM v1** | Bi-Temporal | 8.9M | 0.8290 | 0.7080 | 0.7245 |

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- Jebur et al. (2026). *A hybrid remote sensing socio-ecological index (HRSEI) based on spatial grid GNN*. Journal of Hazardous Materials Advances.
- Hu & Wu (2026). *Venagnosis: A Progressive Fusion Graph Network for Parcel-Level Degradation Mapping*. IEEE Access.
- Liu et al. (2026). *Graph-based Semantic Rerouting Interaction for Rotation-Robust Remote Sensing*. ACM International Conference on Multimedia Retrieval (ICMR).
- Shao et al. (2026). *CGMN: Multi-Branch Spatial Graph Recurrent Network for Forest Canopy Disturbance*. GIScience & Remote Sensing.
- Cheng et al. (2021). *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation*. IEEE/CVF CVPR.
