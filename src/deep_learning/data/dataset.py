import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from deep_learning.config import (
    BATCH_SIZE,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
)
from deep_learning.data.transforms import (
    test_transform,
    train_transform,
    validation_transform,
)

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=test_transform)
validation_dataset = datasets.ImageFolder(
    root=VALIDATION_DATA_DIR, transform=validation_transform
)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count() or 4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)

test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count() or 4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)

validation_data_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count() or 4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)
