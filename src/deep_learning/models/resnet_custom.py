from torch import nn
from torchvision import models
from torchvision.models import ResNet, ResNet50_Weights

from deep_learning.models.base_model import BaseModel


class ResNetCustom(BaseModel):
    model: ResNet
    num_classes: int

    def __init__(self, num_classes: int):
        super().__init__()

        weights: ResNet50_Weights = models.ResNet50_Weights.DEFAULT
        resnet: ResNet = models.resnet50(weights=weights)

        self.model = resnet
        self.num_classes = num_classes

    def switch_head(self):
        original_in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=original_in_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=self.num_classes),
        )

    def get_last_backbone_layer_parameters(self):
        return self.model.layer4.parameters()

    def get_head_parameters(self):
        return self.model.fc.parameters()
