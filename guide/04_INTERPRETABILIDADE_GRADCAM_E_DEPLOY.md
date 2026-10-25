# 🔍 Guia 04: Interpretabilidade com Grad-CAM & Deploy

> **Prioridade:** 🚀 **Produção & Auditoria Técnica**
> **Objetivo:** Auditar decisões visuais de modelos pré-treinados com **Grad-CAM**, construir pipelines de inferência de alta velocidade para aplicações web (Streamlit / FastAPI) e exportar redes para formatos de produção (**ONNX** e **TorchScript**).

---

### 🌿 O Caso AgroVision: A Caixa-Preta do `predictor.py`

No arquivo [`src/deep_learning/engine/predictor.py`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/engine/predictor.py>), construímos a função `predict_single_image` para alimentar a interface Streamlit do AgroVision:

```python
# Trecho do seu src/deep_learning/engine/predictor.py
with torch.no_grad():
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    higher_prob, higher_prob_index = torch.max(probabilities, dim=1)
```

#### Qual é a limitação dessa abordagem para diagnóstico agrícola?

Na interface web, o agrônomo faz o upload da foto e a tela exibe: *"Oídio: 97% de confiança"*.

* Mas **como o usuário sabe se o modelo realmente olhou para as manchas brancas de fungo** ou se ele apenas decorou a cor marrom do vaso ou da bancada ao fundo?
* Em aplicações biomédicas e agronômicas, uma predição sem mapa de calor explicativo é uma **caixa-preta não-auditável**.

---

## 1. Por que Interpretabilidade Visual é Indispensável?

Modelos de Visão Computacional são frequentemente chamados de "caixas-pretas". Se uma rede pré-treinada atinge 98% de acurácia na classificação de doenças foliares, como garantir que ela aprendeu as características da patologia, e não:

* A cor da bancada do laboratório?
* A mão do agrônomo que segurava a folha?
* Um artefato de compressão específico da câmera?

A técnica consagrada na literatura científica para auditar e resolver esse problema no AgroVision é o **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

```
[ Imagem de Entrada ] ──► [ Rede Convolucional ] ──► [ Predição da Classe ]
                                   │                          │
                                   ▼                          ▼
                       [ Ativações Convolucionais ] ◄── [ Gradientes da Classe ]
                                   │
                                   ▼
                       [ Mapa de Calor (Heatmap) ]
```

### O que o mapa de calor nos diz:

* **Áreas em Vermelho / Amarelo (Cores Quentes):** Regiões com maior peso gradiente na decisão. Se coincidirem com as lesões de fungos, o modelo está cientificamente validado.
* **Áreas em Azul (Cores Frias):** Regiões ignoradas pelo modelo.

---

## 2. Implementando o Grad-CAM Estritamente Tipado em PyTorch

O Grad-CAM utiliza **Hooks** do PyTorch para interceptar as ativações e os gradientes no último bloco convolucional (`layer4[-1]` na ResNet ou `features[-1]` na MobileNet).

Para atender aos padrões corporativos de produção e evitar vazamentos de memória (memory leaks na VRAM), a classe implementa tipagem estrita via `typing` / `torch.utils.hooks` e suporte a **Context Manager** (`with`):

```python
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


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
        # [1, 2048, 7, 7] -> mean(dim=(2, 3), keepdim=True) -> [1, 2048, 1, 1]
        weights: torch.Tensor = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Multiplicação ponderada das ativações pelos pesos:
        # [1, 2048, 1, 1] * [1, 2048, 7, 7] = [1, 2048, 7, 7]
        # Soma sobre o canal (dim=1) colapsando os 2048 canais em 1 mapa 2D -> [1, 7, 7]
        weighted_activations: torch.Tensor = torch.sum(weights * self.activations, dim=1)

        # 5. Colapsa a dimensão de batch (dim=0) -> [7, 7]
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
        alpha (float): Fator de opacidade da fusão (0.0 = apenas original, 1.0 = apenas calor).
        colormap_name (str): Nome do mapa de cores do matplotlib ('jet', 'magma', 'viridis').

    Returns:
        Image.Image: Imagem resultante com o mapa de calor sobreposto.
    """
    width, height = original_image.size

    # Interpolação bilinear do heatmap [H_feat, W_feat] para o tamanho da imagem original [H, W]
    heatmap_batch = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H_feat, W_feat]
    heatmap_resized = F.interpolate(
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

    # Fusão suave (blend) entre a foto real da folha e as cores do mapa de calor
    return Image.blend(original_image.convert("RGB"), heatmap_pil, alpha=alpha)
```

### 💡 Exemplo de Uso Prático Passo a Passo

Veja como instanciar a classe, calcular o mapa de calor da folha e exibir o diagnóstico auditado:

```python
from pathlib import Path
from PIL import Image
import torch
from torchvision import models

# 1. Carrega o modelo treinado e o pré-processamento oficial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pesos = models.ResNet50_Weights.DEFAULT
modelo = models.resnet50(weights=pesos).to(device)
transform = pesos.transforms()

# 2. Seleciona a última camada convolucional profunda
# Para ResNet-18/34/50: o último bloco da layer4 (layer4[-1])
# Para MobileNetV3: o último bloco de features (features[-1])
camada_alvo = modelo.layer4[-1]

# 3. Carrega e prepara a imagem da folha
imagem_folha = Image.open("dados/folha_exemplo.jpg").convert("RGB")
# Adiciona a dimensão de batch com .unsqueeze(0) -> [1, 3, 224, 224]
tensor_entrada: torch.Tensor = transform(imagem_folha).unsqueeze(0).to(device)

# 4. Gera o heatmap via Context Manager (registra e remove hooks automaticamente)
with GradCAM(model=modelo, target_layer=camada_alvo) as gradcam:
    # Se omitir target_class, o Grad-CAM explica a classe de maior logit prevista
    heatmap: torch.Tensor = gradcam.generate_heatmap(tensor_entrada)

# 5. Gera a imagem final com o mapa de calor sobreposto à foto original
imagem_auditada: Image.Image = overlay_cam_on_image(
    original_image=imagem_folha,
    heatmap=heatmap,
    alpha=0.45,
    colormap_name="jet",
)

# 6. Salva no disco ou exibe diretamente no Streamlit:
imagem_auditada.save("diagnostico_auditado.png")

# Exemplo de integração no Streamlit (src/deep_learning/ui/pages/predict_single.py):
# col1, col2 = st.columns(2)
# col1.image(imagem_folha, caption="Foto Original", width="stretch")
# col2.image(imagem_auditada, caption="Auditoria Grad-CAM (Onde a rede olhou)", width="stretch")
```

#### 📌 Dica de Arquitetura: Como obter a camada alvo de forma limpa no seu projeto?

Para manter o princípio de encapsulamento no seu AgroVision (`BaseModel`), adicione um método que retorna a camada convolucional profunda:

```python
# No seu src/deep_learning/models/resnet_custom.py:
def get_last_conv_layer(self) -> nn.Module:
    return self.model.layer4[-1]

# No seu src/deep_learning/models/mobilenet_custom.py:
def get_last_conv_layer(self) -> nn.Module:
    return self.model.features[-1]
```

---

## 3. Pipeline de Inferência Tipado para Produção (Streamlit / FastAPI)

Para servir predições em tempo real sem vazamento de memória e com contratos de tipagem bem definidos:

```python
from io import BytesIO
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from PIL import Image


def inferir_imagem(
    caminho_ou_arquivo: str | Path | BytesIO,
    modelo: nn.Module,
    classes: list[str],
    transform: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    Executa a inferência de ponta a ponta com um modelo adaptado pré-treinado.

    Args:
        caminho_ou_arquivo: Caminho em disco (str, Path) ou buffer de upload da web (BytesIO).
        modelo: Rede neural PyTorch treinada.
        classes: Lista ordenada dos nomes das classes.
        transform: Pipeline de transformações (ex: weights.transforms()).
        device: Dispositivo de execução (CPU ou CUDA/ROCm).

    Returns:
        dict[str, Any]: Diagnóstico estruturado com classe vencedora, confiança e probabilidades.
    """
    # 1. Modo de avaliação (desativa Dropout e fixa estatísticas de BatchNorm)
    modelo.eval()

    # 2. Carrega a imagem e garante 3 canais RGB
    imagem: Image.Image = Image.open(caminho_ou_arquivo).convert("RGB")

    # 3. Pré-processa e adiciona dimensão de batch via unsqueeze(0) -> [1, 3, 224, 224]
    tensor_lote: torch.Tensor = transform(imagem).unsqueeze(0).to(device)

    # 4. Desativa gravação de gradientes (economiza VRAM e acelera o tempo de resposta)
    with torch.no_grad():
        logits: torch.Tensor = modelo(tensor_lote)
        probabilidades: np.ndarray = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 5. Obtém a classe de maior probabilidade
    idx_maior: int = int(probabilidades.argmax())

    return {
        "classe_prevista": classes[idx_maior],
        "confianca_percentual": float(probabilidades[idx_maior] * 100),
        "todas_probabilidades": {
            classes[i]: float(probabilidades[i] * 100)
            for i in range(len(classes))
        },
    }
```

---

## 4. Exportação para Produção: ONNX e TorchScript

Quando colocamos redes em ambientes corporativos de alta demanda, executar via Python puro pode introduzir latência devido ao GIL (Global Interpreter Lock). Podemos compilar a rede para formatos otimizados:

### Opção A: TorchScript (Formato Oficial do PyTorch)

```python
modelo.eval()
amostra_dummy = torch.randn(1, 3, 224, 224)

# Compila o grafo computacional via tracing
modelo_compilado = torch.jit.trace(modelo, amostra_dummy)
modelo_compilado.save("modelo_otimizado.pt")

# Pode ser carregado em C++ ou Python sem depender da definição original da classe!
```

### Opção B: ONNX (Open Neural Network Exchange)

```python
torch.onnx.export(
    modelo,
    amostra_dummy,
    "modelo.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

O arquivo `.onnx` pode ser executado em runtimes ultrarrápidos como **ONNX Runtime**, **TensorRT** (NVIDIA) ou compiladores de borda para celulares e microcontroladores.
