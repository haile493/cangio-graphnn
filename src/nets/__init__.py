"""Self-contained model implementations for the Can Gio graph models (no external model library required).

Importing this package registers all architectures and losses in `MODEL_REGISTRY` / `LOSS_REGISTRY`.
"""

from src.nets import big_sam, big_sam_v2, graph_models, losses  # noqa: F401  (registration side effects)
from src.nets.base import BaseModel, ModelConfig, ModelOutput
from src.nets.big_sam import BiGSAMConfig, BiGSAMLoss, BiGSAMNet, BinaryFocalLoss
from src.nets.big_sam_v2 import BiGSAMLossV2, BiGSAMNetV2, BiGSAMV2Config
from src.nets.graph_models import CGMNMultiBranchNet, GSRIRotationRobustNet, HRSEIGNNNet, VenagnosisPFGNNNet
from src.nets.losses import BinaryDiceLoss, CGMNCompositeLoss, FocalLoss, VenagnosisLoss
from src.nets.registry import LOSS_REGISTRY, MODEL_REGISTRY, build_loss, build_model

__all__ = [
    "BaseModel", "ModelConfig", "ModelOutput",
    "BiGSAMConfig", "BiGSAMLoss", "BiGSAMNet", "BinaryFocalLoss",
    "BiGSAMLossV2", "BiGSAMNetV2", "BiGSAMV2Config",
    "CGMNMultiBranchNet", "GSRIRotationRobustNet", "HRSEIGNNNet", "VenagnosisPFGNNNet",
    "BinaryDiceLoss", "CGMNCompositeLoss", "FocalLoss", "VenagnosisLoss",
    "LOSS_REGISTRY", "MODEL_REGISTRY", "build_loss", "build_model",
]
