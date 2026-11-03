"""
PyTorch Dataset and DataLoader Wrappers for PREDATOR NET.

Per Spec §12, §14:
  - Uses float32 throughout.
  - Labels returned as (1,) float32 tensor suitable for BCEWithLogitsLoss.
  - Supports standard NumPy arrays, PyTorch tensors, or memory-mapped files (.npy).
"""

from typing import Union, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FlowDataset(Dataset):
    """
    High-performance PyTorch Dataset for network flow records.
    """

    def __init__(
        self,
        X: Union[np.ndarray, str, Path],
        y: Union[np.ndarray, str, Path],
        mmap_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            X: Feature array or path to .npy file.
            y: Label array or path to .npy file.
            mmap_mode: "r" if memory mapping is requested, or None for in-memory.
        """
        # Load features
        if isinstance(X, (str, Path)):
            self.X = np.load(str(X), mmap_mode=mmap_mode)
        else:
            self.X = np.asarray(X, dtype=np.float32)

        # Load labels
        if isinstance(y, (str, Path)):
            self.y = np.load(str(y), mmap_mode=mmap_mode)
        else:
            self.y = np.asarray(y, dtype=np.float32)

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Feature length ({len(self.X)}) does not match label length ({len(self.y)})"
            )

        # Ensure label shape is (N, 1) float32
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1).astype(np.float32)
        else:
            self.y = self.y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_val = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y_val = torch.from_numpy(np.array(self.y[idx], dtype=np.float32))
        return x_val, y_val


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Factory function to construct an optimized DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )
