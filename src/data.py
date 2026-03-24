from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, bld: np.ndarray, stimulus: np.ndarray, action: np.ndarray, bldi: np.ndarray):
        self.bld = torch.tensor(bld, dtype=torch.float32)
        self.stimulus = torch.tensor(stimulus, dtype=torch.float32)
        self.action = torch.tensor(action, dtype=torch.float32)
        self.bldi = torch.tensor(bldi, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.bld)

    def __getitem__(self, index: int):
        return self.bld[index], self.stimulus[index], self.action[index], self.bldi[index]


def load_npz(dataset_path: Path) -> dict[str, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as npz_file:
        return {key: npz_file[key] for key in npz_file.files}


def load_split_dataset(dataset_path: Path, split: str) -> ArrayDataset:
    arrays = load_npz(dataset_path)
    return ArrayDataset(
        arrays[f"{split}_bld"],
        arrays[f"{split}_stimulus"],
        arrays[f"{split}_action"],
        arrays[f"{split}_bldi"],
    )
