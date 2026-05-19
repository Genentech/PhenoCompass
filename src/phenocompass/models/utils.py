"""Model utilities."""

import collections
import os
from typing import List, Optional


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
