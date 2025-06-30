import os
from collections.abc import Callable

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class EnvironmentDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable[..., any] | None = None,
        split: str = "train",
        val_ratio: float = 0.1,
    ) -> None:
        assert split in ["train", "val"]
        self.img_list = [f for f in os.listdir(root)]
        self.len_all = len(self.img_list)
        self.data = self.img_list[:int(self.len_all * (1-val_ratio))] if split == "train" else self.img_list[int(self.len_all * val_ratio) :]
        self.transform = transform
        self.root = root

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> any:
        img_path = self.root + self.data[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

