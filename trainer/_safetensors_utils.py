"""Minimal safetensors utilities (in-housed from sd-scripts).

Provides only what traintrain needs: load_safetensors and WeightTransformHooks.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from safetensors.torch import load_file

from trainer._device_utils import synchronize_device


# ---------------------------------------------------------------------------
# WeightTransformHooks — used by anima_utils for key renaming during model load
# ---------------------------------------------------------------------------

@dataclass
class WeightTransformHooks:
    split_hook: Optional[Callable] = None
    concat_hook: Optional[Callable] = None
    rename_hook: Optional[Callable] = None


# ---------------------------------------------------------------------------
# MemoryEfficientSafeOpen — used by load_safetensors(disable_mmap=True)
# ---------------------------------------------------------------------------

class MemoryEfficientSafeOpen:
    """Memory-efficient safetensors reader (no full mmap)."""

    def __init__(self, path: str, disable_numpy_memmap: bool = False):
        self.path = path
        self.disable_numpy_memmap = disable_numpy_memmap
        self._data = None
        self._keys = None

    def __enter__(self):
        import json
        import struct
        # Read the file header to get tensor metadata
        with open(self.path, "rb") as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_bytes = f.read(header_len)
            self._header = json.loads(header_bytes.decode("utf-8"))
            self._data_start = 8 + header_len
            self._keys = [k for k in self._header.keys() if k != "__metadata__"]
        return self

    def __exit__(self, *args):
        self._data = None
        self._keys = None

    def keys(self):
        return self._keys

    def get_tensor(self, key: str, device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        import numpy as np
        info = self._header[key]
        offset = self._data_start + info["data_offsets"][0]
        length = info["data_offsets"][1] - info["data_offsets"][0]
        with open(self.path, "rb") as f:
            f.seek(offset)
            tensor_bytes = f.read(length)
        dtype_str = info["dtype"]
        if dtype_str == "BF16" and not hasattr(np, "bfloat16"):
            # NumPy < 2.0 doesn't have bfloat16; read as uint16 and convert via torch
            arr = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(info["shape"])
            tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16)
        else:
            np_dtype = self._dtype_to_numpy(dtype_str)
            arr = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(info["shape"])
            tensor = torch.from_numpy(arr.copy())
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def _dtype_to_numpy(dtype_str: str) -> np.dtype:
        # np.bfloat16 was added in NumPy 2.0; for older NumPy, BF16 is handled
        # directly in get_tensor() before this method is called.
        _BF16 = getattr(np, "bfloat16", np.uint16)
        mapping = {
            "F64": np.float64, "F32": np.float32, "F16": np.float16,
            "BF16": _BF16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
            "U8": np.uint8, "BOOL": bool,
        }
        return mapping[dtype_str]


# ---------------------------------------------------------------------------
# load_safetensors — main entry point for loading .safetensors files
# ---------------------------------------------------------------------------

def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> Dict[str, torch.Tensor]:
    if disable_mmap:
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except Exception:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def get_split_weight_filenames(file_path: str) -> Optional[List[str]]:
    """Get the list of split weight filenames if the file name ends with 00001-of-00004 etc."""
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        filenames = []
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} not found")
        return filenames
    else:
        return None
