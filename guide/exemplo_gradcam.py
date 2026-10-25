"""
exemplo_gradcam.py
Demonstração prática e estritamente tipada do Grad-CAM com TorchVision:
1. Carregamento de arquitetura com pesos pré-treinados (ResNet-18 ou ResNet-50)
2. Registro seguro de Hooks em camadas convolucionais com RemovableHandle
3. Interceptação de ativações (forward) e gradientes da classe alvo (backward)
4. Redução dimensional e ponderação por canal (Global Average Pooling)
5. Superposição do mapa de calor sobre a imagem via Matplotlib e PIL (sem OpenCV)
"""

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from torchvision import models


class GradCAM:
    """
    Implementação estritamente tipada do Grad-CAM (Gradient-weighted Class Activation Mapping).

    Atributos:
        model (nn.Module): Arquitetura de rede neural avaliada.
        target_layer (nn.Module): Camada convolucional profunda inspecionada.
        gradients (torch.Tensor | None): Tensores de gradientes dY/dA interceptados no backward.
        activations (torch.Tensor | None): Mapas de ativação A interceptados no forward.
        hooks (list[RemovableHandle]): Handles para desregistro seguro dos hooks do PyTorch.
    """
    model: nn.Module
    target_layer: nn.Module
    gradients: torch.Tensor | None
    activations: torch.Tensor | None
    hooks: list[RemovableHandle]

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registra hooks no forward e backward da camada convolucional alvo."""
        self.hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Captura os mapas de características da saída da camada (Forward Hook)."""
        self.activations = output

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Captura os gradientes da perda em relação à saída da camada (Backward Hook)."""
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """
        Calcula o mapa de ativação ponderado por gradientes (Grad-CAM).

        Args:
            input_tensor (torch.Tensor): Tensor de entrada no formato [1, C, H, W].
            target_class (int | None): Índice da classe alvo. Se None, utiliza a classe de maior logit.

        Returns:
            torch.Tensor: Mapa 2D de relevância no formato [H_feat, W_feat] normalizado em [0.0, 1.0].
        """
        self.model.eval()

        # 1. Forward pass capturando ativações -> logits: [1, num_classes]
        logits: torch.Tensor = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # 2. Zera gradientes prévios e calcula backward específico para a classe alvo
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Falha ao capturar ativações ou gradientes via hooks do PyTorch.")

        # 3. Global Average Pooling nos gradientes espaciais:
        # [1, C, H, W] -> mean(dim=(2, 3), keepdim=True) -> [1, C, 1, 1]
        weights: torch.Tensor = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Multiplicação ponderada das ativações pelos pesos:
        # [1, C, 1, 1] * [1, C, H, W] = [1, C, H, W]
        # Soma sobre o canal (dim=1) colapsando canais em 1 mapa 2D -> [1, H, W]
        weighted_activations: torch.Tensor = torch.sum(weights * self.activations, dim=1)

        # 5. Colapsa a dimensão de batch (dim=0) -> [H, W]
        cam: torch.Tensor = weighted_activations.squeeze(0)

        # 6. ReLU: retém apenas os pixels com correlação positiva para a classe
        cam = F.relu(cam)

        # 7. Normalização Min-Max para a escala contínua [0.0, 1.0]
        cam_min: torch.Tensor = cam.min()
        cam_max: torch.Tensor = cam.max()
        cam_normalized: torch.Tensor = (cam - cam_min) / ((cam_max - cam_min) + 1e-8)

        return cam_normalized.detach()

    def remove_hooks(self) -> None:
        """Remove os hooks registrados para liberar memória de VRAM e evitar retenção de grafos."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.remove_hooks()


def overlay_cam_on_image(
    original_image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Superpõe o mapa de calor Grad-CAM sobre a imagem original em PIL sem dependência do OpenCV.

    Args:
        original_image (Image.Image): Imagem original carregada via PIL (RGB).
        heatmap (torch.Tensor): Mapa de calor bruto 2D [H_feat, W_feat] em [0, 1].
        alpha (float): Fator de opacidade da fusão (0.0 = original pura, 1.0 = calor puro).
        colormap_name (str): Nome do mapa de cores do matplotlib ('jet', 'magma', 'viridis').

    Returns:
        Image.Image: Imagem com o mapa de calor sobreposto.
    """
    width, height = original_image.size

    # Interpolação bilinear do mapa [H_feat, W_feat] para o tamanho da imagem original [H, W]
    heatmap_batch: torch.Tensor = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H_feat, W_feat]
    heatmap_resized: np.ndarray = F.interpolate(
        heatmap_batch,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Aplica colormap do matplotlib (retorna matriz RGBA normalizada [0.0, 1.0])
    cmap = plt.get_cmap(colormap_name)
    colored_cam_rgb: np.ndarray = cmap(heatmap_resized)[:, :, :3]  # descarta canal alpha

    # Converte matriz float [0, 1] para imagem PIL uint8 [0, 255]
    heatmap_pil: Image.Image = Image.fromarray(np.uint8(colored_cam_rgb * 255))

    # Fusão suave (blend) entre a foto real e as cores do mapa de calor
    return Image.blend(original_image.convert("RGB"), heatmap_pil, alpha=alpha)


def demonstrar_gradcam() -> None:
    print("=" * 65)
    print("DEMONSTRAÇÃO ESTREITAMENTE TIPADA DO GRAD-CAM (PYTORCH)")
    print("=" * 65)

    # 1. Instancia modelo ResNet-18 para demonstração rápida
    pesos = models.ResNet18_Weights.DEFAULT
    modelo: nn.Module = models.resnet18(weights=pesos)
    transform = pesos.transforms()

    # Última camada convolucional profunda da ResNet-18:
    camada_alvo: nn.Module = modelo.layer4[-1]
    print(f"✓ Modelo carregado: ResNet-18")
    print(f"✓ Camada alvo para hooks: {camada_alvo.__class__.__name__} (layer4[-1])")

    # 2. Cria imagem sintética simulando uma folha com mancha patológica
    imagem_base: Image.Image = Image.new("RGB", (224, 224), color=(34, 139, 34))  # Verde
    tensor_entrada: torch.Tensor = transform(imagem_base).unsqueeze(0)  # [1, 3, 224, 224]

    print(f"✓ Imagem de teste convertida para tensor: {list(tensor_entrada.shape)}")

    # 3. Execução auditada do Grad-CAM via Context Manager
    with GradCAM(model=modelo, target_layer=camada_alvo) as gradcam:
        print("✓ Hooks registrados com sucesso.")
        heatmap: torch.Tensor = gradcam.generate_heatmap(tensor_entrada)
        print(f"✓ Heatmap gerado:")
        print(f"    - Shape bruto: {list(heatmap.shape)} (dimensão da grade convolucional)")
        print(f"    - Faixa de valores: min={heatmap.min().item():.3f}, max={heatmap.max().item():.3f}")

    print("✓ Hooks desregistrados automaticamente na saída do bloco 'with'.")

    # 4. Superposição do mapa de calor sobre a imagem original
    resultado_overlay: Image.Image = overlay_cam_on_image(
        original_image=imagem_base,
        heatmap=heatmap,
        alpha=0.45,
        colormap_name="jet",
    )
    print(f"✓ Mapa de calor interpolado e fundido com a imagem original:")
    print(f"    - Dimensões finais (PIL): {resultado_overlay.size}")
    print("\n" + "=" * 65)
    print("✓ PIPELINE DE INTERPRETABILIDADE 100% OPERACIONAL.")
    print("=" * 65)


if __name__ == "__main__":
    demonstrar_gradcam()
