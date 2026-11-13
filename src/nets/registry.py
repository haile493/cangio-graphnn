"""
Hierarchical Component Registry for Extensible Model & Module Construction.

Allows seamless registration and configuration-driven instantiation of:
- Complete Models (`MODEL_REGISTRY`)
- Architectural Blocks (`BLOCK_REGISTRY`)
- Attention & Context Modules (`ATTENTION_REGISTRY`)
- Task Heads (`HEAD_REGISTRY`)
- Loss Functions (`LOSS_REGISTRY`)
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union


class Registry:
    """
    A generic, durable registry mapping string keys (and optional aliases)
    to classes or factory functions.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Union[Type[Any], Callable[..., Any]]] = {}

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._module_dict

    def __repr__(self) -> str:
        items = ", ".join(sorted(self._module_dict.keys()))
        return f"Registry(name={self._name!r}, items=[{items}])"

    def keys(self) -> List[str]:
        return sorted(self._module_dict.keys())

    def get(self, key: str) -> Union[Type[Any], Callable[..., Any]]:
        norm_key = key.lower()
        if norm_key not in self._module_dict:
            available = ", ".join(self.keys())
            raise KeyError(
                f"'{key}' is not registered in {self._name} registry. "
                f"Available items: [{available}]"
            )
        return self._module_dict[norm_key]

    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        override: bool = True,
    ) -> Callable[[Union[Type[Any], Callable[..., Any]]], Union[Type[Any], Callable[..., Any]]]:
        """
        Decorator to register a class or builder function.

        Example:
            @MODEL_REGISTRY.register("waveco_mtl", aliases=["waveco_mtl_3d"])
            class WaveCoMTLNet(BaseModel):
                ...
        """
        def _register(cls_or_fn: Union[Type[Any], Callable[..., Any]]):
            primary_name = (name or cls_or_fn.__name__).lower()
            all_names = [primary_name]
            if aliases:
                all_names.extend([a.lower() for a in aliases])

            for k in all_names:
                if not override and k in self._module_dict:
                    raise KeyError(f"'{k}' is already registered in {self._name}.")
                self._module_dict[k] = cls_or_fn
            return cls_or_fn

        return _register

    def build(self, name_or_cfg: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
        """
        Instantiates a registered component by string name or config dictionary.

        Args:
            name_or_cfg: Either the registered name (str) or a dict containing a ``"type"``
                or ``"name"`` key along with constructor arguments.
            **kwargs: Additional keyword arguments overriding or supplementing config.
        """
        if isinstance(name_or_cfg, str):
            target_cls = self.get(name_or_cfg)
            return target_cls(**kwargs)
        elif isinstance(name_or_cfg, dict):
            cfg = dict(name_or_cfg)
            type_key = cfg.pop("type", None) or cfg.pop("name", None)
            if not type_key:
                raise ValueError(
                    f"Config dict passed to {self._name}.build() must contain 'type' or 'name'."
                )
            cfg.update(kwargs)
            target_cls = self.get(type_key)
            return target_cls(**cfg)
        else:
            raise TypeError(
                f"Expected str or dict for {self._name}.build(), got {type(name_or_cfg).__name__}"
            )


# Global singleton registries
MODEL_REGISTRY = Registry("models")
BLOCK_REGISTRY = Registry("blocks")
ATTENTION_REGISTRY = Registry("attentions")
HEAD_REGISTRY = Registry("heads")
LOSS_REGISTRY = Registry("losses")


def build_model(name_or_cfg: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
    """Convenience builder for any registered model."""
    return MODEL_REGISTRY.build(name_or_cfg, **kwargs)


def build_block(name_or_cfg: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
    """Convenience builder for any registered architectural block."""
    return BLOCK_REGISTRY.build(name_or_cfg, **kwargs)


def build_attention(name_or_cfg: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
    """Convenience builder for any registered attention module."""
    return ATTENTION_REGISTRY.build(name_or_cfg, **kwargs)


def build_loss(name_or_cfg: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
    """Convenience builder for any registered loss function."""
    return LOSS_REGISTRY.build(name_or_cfg, **kwargs)


def list_models() -> List[str]:
    """Lists all registered model keys."""
    return MODEL_REGISTRY.keys()
