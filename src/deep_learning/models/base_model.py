from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import Parameter


class BaseModel(nn.Module, ABC):
    model: nn.Module
    num_classes: int

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def freeze_gradients(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze_last_backbone_layer(self):
        for parameter in self.get_last_backbone_layer_parameters():
            parameter.requires_grad = True

    @abstractmethod
    def switch_head(self) -> None:
        pass

    @abstractmethod
    def get_last_backbone_layer_parameters(self) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def get_head_parameters(self) -> Iterator[Parameter]:
        pass
