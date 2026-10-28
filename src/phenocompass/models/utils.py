"""Model utilities."""

import collections
import inspect
import os
import sys
import types
from typing import List, Optional

# Released checkpoints were trained under the package's former name and pickle
# nn.Module hyperparameters from this path.
_LEGACY_COMPONENTS_MODULE = "multimodal_contrastive.networks.components"


def register_legacy_module_aliases():
    """Alias the pre-release ``multimodal_contrastive`` package to ``phenocompass``.

    The released PhenoCompass checkpoints store encoder modules in their
    hyperparameters under ``multimodal_contrastive.networks.components``.
    Mapping that path onto ``phenocompass.models.components`` (same class
    names and attributes) lets them unpickle without the original package.
    """
    from phenocompass.models import components

    parts = _LEGACY_COMPONENTS_MODULE.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault(_LEGACY_COMPONENTS_MODULE, components)


def load_trusted_checkpoint(module_cls, checkpoint_path, **kwargs):
    """Load a released PhenoCompass/GNEprop Lightning checkpoint.

    The checkpoints pickle omegaconf, typing and nn.Module objects, which
    PyTorch >= 2.6 rejects under its ``weights_only=True`` default. Only use
    this for checkpoints from a trusted source (e.g. the PhenoCompass Zenodo
    release), since full unpickling can execute arbitrary code.

    Parameters
    ----------
    module_cls : type
        LightningModule subclass providing ``load_from_checkpoint``.
    checkpoint_path : str
        Path to the ``.ckpt`` file.
    **kwargs
        Forwarded to ``load_from_checkpoint`` (e.g. ``map_location``).

    Returns
    -------
    LightningModule
        The loaded model.
    """
    register_legacy_module_aliases()
    if "weights_only" in inspect.signature(module_cls.load_from_checkpoint).parameters:
        kwargs.setdefault("weights_only", False)
    return module_cls.load_from_checkpoint(checkpoint_path, **kwargs)


def move_batch_input_to_device(x, device):
    """Recursively move batch inputs to device.

    Handles nested dictionaries and lists of tensors.

    Parameters
    ----------
    x : dict
        Dictionary of inputs, potentially nested.
    device : torch.device or str
        Target device.

    Returns
    -------
    dict
        Input dictionary with all tensors moved to device.
    """
    for key, item in x.items():
        if isinstance(item, collections.abc.Mapping):
            x[key] = move_batch_input_to_device(x.get(key, {}), device)
        elif isinstance(item, list):
            x[key] = [v.to(device) for v in item]
        else:
            x[key] = item.to(device)
    return x


def get_checkpoint_paths(
    checkpoint_path: Optional[str] = None,
    checkpoint_paths: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None,
    ext: str = ".ckpt",
) -> Optional[List[str]]:
    """Get a list of checkpoint paths.

    Gets checkpoint paths either from a single checkpoint path, a list of paths,
    or from a directory of checkpoints.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to a single checkpoint.
    checkpoint_paths : list of str, optional
        List of paths to checkpoints.
    checkpoint_dir : str, optional
        Path to a directory containing checkpoints.
    ext : str, default ".ckpt"
        The extension which defines a checkpoint file.

    Returns
    -------
    list of str or None
        A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.

    Raises
    ------
    ValueError
        If more than one of checkpoint_dir, checkpoint_path, checkpoint_paths is provided,
        or if no checkpoints are found in the specified directory.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError(
            "Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths"
        )

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(
                f'Failed to find any checkpoints with extension "{ext}" '
                f'in directory "{checkpoint_dir}"'
            )

        return checkpoint_paths

    return None
