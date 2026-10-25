from typing import Self
from typing import Any
from torch.utils.hooks import RemovableHandle
import torch
from torch import nn
import torch.nn.functional as F


class GradCAM:
    model: nn.Module
    target_layer: nn.Module
    gradients: torch.Tensor | None
    activations: torch.Tensor | None
    hooks: list[RemovableHandle]

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self.register_hooks()

    def register_hooks(self) -> None:
        self.hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ):
        self.activations = output

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ):
        self.model.eval()

        logits: torch.Tensor = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()

        logits[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Falha ao capturar ativações ou gradientes")

        weights: torch.Tensor = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations: torch.Tensor = torch.sum(
            weights * self.activations, dim=1
        )

        cam: torch.Tensor = weighted_activations.squeeze(0)
        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        cam_normalized = (cam - cam_min) / ((cam_max - cam_min) + 1e-8)

        return cam_normalized.detach()

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.remove_hooks()
