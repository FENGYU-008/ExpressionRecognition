# -*- encoding: utf-8 -*-
import os
from typing import Tuple, Any, Callable, Optional

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
import csv


class FER2013(VisionDataset):
    """
    FER2013 format:
        index   emotion     pixels      Usage

    index: id of series
    emotion: label (from 0 - 6)
    pixels: 48x48 pixel value (uint8)
    Usage: [Training, PrivateTest, PublicTest]
    """

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

        data_file = os.path.join(root, 'fer2013', split + ".csv")

        with open(data_file, "r", newline="") as file:
            self._samples = [
                (
                    torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                    int(row["emotion"]) if "emotion" in row else None,
                )
                for row in csv.DictReader(file)
            ]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self._samples)
