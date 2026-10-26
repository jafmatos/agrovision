from typing import Any, Self

import numpy
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch import nn
from torch.utils.hooks import RemovableHandle


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
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        colormap_name: str = "jet",
    ) -> numpy.ndarray:
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
        cam_normalized = cam_normalized.detach()

        height, width = list(input_tensor.shape)[-2:]

        heatmap_batch = cam_normalized.unsqueeze(0).unsqueeze(0)
        heatmap_resized = (
            F.interpolate(
                heatmap_batch,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        cmap = pyplot.get_cmap(colormap_name)
        colored_cam = cmap(heatmap_resized)[:, :, :3]

        return numpy.uint8(colored_cam * 255)

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
