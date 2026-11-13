"""
Base Model Contracts, Configuration Containers, and Standardized Output Wrappers.

Ensures durability and a uniform interface across 2D, 3D, Spatiotemporal, Multi-Task,
and Sequential models while maintaining backward compatibility with dict-based scripts.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    Standardized, serializable configuration container for building models.
    Supports arbitrary extension parameters via `extra`.
    """
    arch: str = "waveco_mtl"
    dims: int = 3
    in_channels: int = 1
    num_seg_classes: int = 2
    num_cls_classes: int = 5
    base_channels: int = 24
    norm_type: str = "instancenorm"
    act_type: str = "leakyrelu"
    dropout: float = 0.2
    pretrained: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        init_kwargs = {k: v for k, v in data.items() if k in known_fields}
        extra_kwargs = dict(data.get("extra", {}))
        for k, v in data.items():
            if k not in known_fields and k != "extra":
                extra_kwargs[k] = v
        return cls(**init_kwargs, extra=extra_kwargs)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class ModelOutput(dict):
    """
    Hybrid Dictionary & Attribute-Accessible Model Output Container.

    Allows both:
    - Attribute access: `out.seg_logits`, `out.cls_logits`, `out.latent_feat`
    - Dict access (100% backward-compatible with all existing project scripts):
      `out["seg_logits"]`, `'cls_logits' in out`, `out.get('aux_seg_logits')`
    """

    def __init__(
        self,
        seg_logits: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        aux_seg_logits: Optional[torch.Tensor] = None,
        edge_logits: Optional[torch.Tensor] = None,
        latent_feat: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if seg_logits is not None:
            self["seg_logits"] = seg_logits
        if cls_logits is not None:
            self["cls_logits"] = cls_logits
        if aux_seg_logits is not None:
            self["aux_seg_logits"] = aux_seg_logits
        if edge_logits is not None:
            self["edge_logits"] = edge_logits
        if latent_feat is not None:
            self["latent_feat"] = latent_feat
        for k, v in kwargs.items():
            if v is not None:
                self[k] = v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all standardized models in `ag_models`.

    Provides built-in utilities for:
    - Parameter counting (trainable vs. total)
    - Architecture summary inspection
    - Checkpointing with metadata and configuration persistence
    - Selective freezing/unfreezing of submodules
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig(arch=self.__class__.__name__)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Union[ModelOutput, torch.Tensor, Tuple[Any, ...]]:
        """Forward pass implemented by concrete model subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, *args: Any, **kwargs: Any) -> Union[ModelOutput, torch.Tensor, Tuple[Any, ...]]:
        """Evaluation-mode inference helper with `torch.no_grad()`."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(*args, **kwargs)
        finally:
            if was_training:
                self.train()

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Returns total or trainable parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> Dict[str, Union[int, float, str]]:
        """Returns structured parameter statistics."""
        total = self.num_parameters(trainable_only=False)
        trainable = self.num_parameters(trainable_only=True)
        ratio = (trainable / total * 100.0) if total > 0 else 0.0
        size_mb = total * 4 / (1024 ** 2)
        return {
            "model_name": self.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(ratio, 2),
            "approx_fp32_size_mb": round(size_mb, 2),
        }

    def freeze_module(self, module_or_name: Union[nn.Module, str]) -> None:
        """Freezes parameters of a target submodule."""
        target = getattr(self, module_or_name) if isinstance(module_or_name, str) else module_or_name
        for p in target.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters in the model."""
        for p in self.parameters():
            p.requires_grad = True

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves model weights alongside config, optimizer state, and training metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: Dict[str, Any] = {
            "model_class": self.__class__.__name__,
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict() if self.config else {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = epoch
        if metrics is not None:
            payload["metrics"] = metrics
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads weights from either a standardized checkpoint dict (`model_state_dict` / `state_dict`)
        or a raw `state_dict` file saved by legacy scripts.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"], strict=strict)
                return ckpt
            elif "state_dict" in ckpt:
                self.load_state_dict(ckpt["state_dict"], strict=strict)
                return ckpt
        self.load_state_dict(ckpt, strict=strict)
        return {"model_state_dict": ckpt}
