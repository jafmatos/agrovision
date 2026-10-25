import torch
from torchvision import models
from torchvision.transforms import v2

from deep_learning.config import IMAGE_SIZE

train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

test_transform = models.ResNet50_Weights.DEFAULT.transforms()

validation_transform = models.ResNet50_Weights.DEFAULT.transforms()
