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

## 4. Exportação para Produção: Compilando para TorchScript e ONNX

No ambiente de desenvolvimento do AgroVision, executamos a rede neural via **PyTorch Eager Mode** através do arquivo `predictor.py`:
```python
# Abordagem de desenvolvimento (Eager Mode)
modelo = carregar_pesos("meu_modelo.pth")
com torch.no_grad():
    saida = modelo(tensor)
```

Embora ideal para experimentação iterativa, o deploy em produção dessa forma introduz **gargalos graves de engenharia**:

1. **Dependência Forte do Código-Fonte Python:** O arquivo `.pth` salva apenas o `state_dict` (dicionário de tensores com pesos e vieses). Para instanciar o modelo em qualquer microsserviço, a definição exata da classe em Python (`class ResNetCustom(nn.Module)`) precisa estar instalada e acessível no ambiente.
2. **Imagens Docker Gigantescas e Cold-Starts Lentos:** A imagem base precisa conter a biblioteca PyTorch inteira (~2.5 GB a 5 GB com drivers CUDA), gerando alto consumo de armazenamento e lentidão crítica em escalonamento automático de contêineres (*auto-scaling*).
3. **Overhead do GIL e Despacho Python:** No CPython, toda operação passa pelo Global Interpreter Lock (GIL) e pela camada de despacho (*dispatch overhead*) do interpretador a cada chamada de camada convolucional, limitando o throughput de requisições simultâneas em CPU.

Para contornar essas limitações, compilamos o modelo para formatos otimizados de grafo estático: **TorchScript** e **ONNX**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Modelo PyTorch Treinado                         │
│                           (ResNet / MobileNet)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│     Opção 1: TorchScript (.pt)    │ │        Opção 2: ONNX (.onnx)      │
│  • Compilação em C++ (LibTorch)   │ │  • Formato aberto e interoperável │
│  • Independente de código Python  │ │  • Sem PyTorch em produção        │
│  • Otimização e fusão de nós      │ │  • Runtime minúsculo (~50 MB)     │
└─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                  │                                     │
                  ▼                                     ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│     Deploy em C++, TorchServe     │ │      ONNX Runtime / TensorRT      │
│      ou Microsserviço Python      │ │      Edge / Web / Servidores      │
└───────────────────────────────────┘ └───────────────────────────────────┘
```

---

### 4.1 TorchScript: O Compilador Nativo do PyTorch

O **TorchScript** é uma representação intermediária (*Intermediate Representation* - IR) do PyTorch serializada em arquivo único. Uma vez exportado, o modelo pode ser executado em processos **C++ puros (sem interpretador Python instalado)** ou em Python livre da contenção do GIL.

Existem duas estratégias para gerar um artefato TorchScript:

#### A) Tracing (`torch.jit.trace`) — O Padrão para Redes de Visão Computacional
Alimenta um tensor arbitrário (*dummy input*) através do modelo e grava a sequência exata de operações executadas pelos tensores.
* **Vantagens:** Funciona imediatamente com quase todas as arquiteturas convolucionais do TorchVision (ResNet, ConvNeXt, MobileNet, EfficientNet).
* **Limitação:** Não captura estruturas de controle dependentes de valores dinâmicos dos tensores (ex.: `if tensor.sum() > 0:`). O tracing congela apenas o caminho executado pelo *dummy input*.

#### B) Scripting (`torch.jit.script`) — Para Fluxos Dinâmicos
Inspeciona diretamente o código-fonte Python e compila a sintaxe para o dialeto TorchScript.
* **Vantagens:** Preserva condicionais (`if/else`) e laços (`for/while`) dependentes do conteúdo dos tensores.
* **Limitação:** Exige que todo o código Python siga um subconjunto estrito de tipagem e sintaxe suportado pelo compilador TorchScript.

#### 🛠️ Código Completo de Exportação para TorchScript

> [!IMPORTANT]
> **Sempre chame `model.eval()` antes da exportação!**  
> Se o modelo estiver em modo de treino (`model.train()`), camadas de `Dropout` gravarão nós estocásticos aleatórios no grafo e camadas de `BatchNorm` continuarão atualizando estatísticas de média e variância durante o rastreamento, corrompendo a inferência.

```python
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def exportar_para_torchscript(
    modelo: nn.Module,
    caminho_saida: str | Path,
    tamanho_entrada: tuple[int, int] = (224, 224),
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Exporta uma arquitetura PyTorch para grafo estático TorchScript via Tracing.

    Args:
        modelo: Rede neural com pesos ajustados.
        caminho_saida: Destino do arquivo compilado (.pt).
        tamanho_entrada: Resolução espacial da imagem (H, W).
        device: Dispositivo onde a compilação será executada.
    """
    # 1. Trava Dropout e fixa estatísticas de BatchNorm
    modelo.eval().to(device)

    # 2. Cria tensor dummy representativo com formato [Batch, Canais, Altura, Largura]
    dummy_input = torch.randn(1, 3, *tamanho_entrada, dtype=torch.float32, device=device)

    # 3. Rastreia o grafo computacional
    with torch.no_grad():
        modelo_traced = torch.jit.trace(modelo, dummy_input)

    # 4. Salva o artefato serializado no disco
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    modelo_traced.save(str(caminho_saida))
    print(f"✅ Modelo TorchScript exportado com sucesso: {caminho_saida}")


# Como inferir usando o modelo TorchScript carregado (sem a classe original!):
def inferir_torchscript(caminho_modelo: str | Path, tensor_entrada: torch.Tensor) -> torch.Tensor:
    # torch.jit.load não requer importação de ResNetCustom ou resnet_custom.py
    modelo_carregado = torch.jit.load(str(caminho_modelo))
    modelo_carregado.eval()
    with torch.no_grad():
        logits = modelo_carregado(tensor_entrada)
    return logits
```

---

### 4.2 ONNX: O Padrão Aberto Universal da Indústria

O **ONNX (Open Neural Network Exchange)** é um formato aberto baseado em *Protocol Buffers* criado para permitir a portabilidade total de modelos entre frameworks (PyTorch, TensorFlow, Scikit-Learn) e motores de inferência de altíssimo desempenho (**ONNX Runtime**, **NVIDIA TensorRT**, **Intel OpenVINO**, **Apple CoreML**).

#### Por que o ONNX é a escolha predileta para microsserviços?
* **Adeus Dependência de PyTorch:** Em servidores de produção, o microsserviço (ex.: FastAPI) precisa apenas do pacote `onnxruntime` (~50 MB), descartando os ~3 GB do PyTorch.
* **Otimização de Grafo Automática:** O runtime aplica fusão de camadas (*Constant Folding*, *Conv+BatchNorm+ReLU fusion*) e vetorização via instruções AVX-512 ou Tensor Cores.
* **Interoperabilidade Total:** O mesmo arquivo `.onnx` pode ser servido em uma API Linux em nuvem, em um aplicativo mobile iOS/Android ou diretamente no navegador web via WebAssembly/WebGPU.

#### ⚠️ A Armadilha dos Eixos Dinâmicos (`dynamic_axes`)

Ao exportar com `torch.onnx.export`, o PyTorch utiliza a dimensão do `dummy_input` para definir a assinatura do grafo. Se o `dummy_input` for `[1, 3, 224, 224]` e você **não** declarar `dynamic_axes`, o grafo aceitará **exclusivamente imagens unitárias**.

Se um lote de 8 imagens for enviado para o modelo em produção, a aplicação sofrerá um erro crítico irrecuperável:
```
RuntimeError: ONNX Runtime failed: Got invalid dimensions for input: expected 1, got 8
```

Para garantir que a API suporte desde 1 imagem até grandes lotes dinâmicos, definimos o eixo 0 como `batch_size`.

#### 🛠️ Código Completo de Exportação para ONNX

> [!NOTE]
> **Pré-requisito para exportação no PyTorch:**  
> O PyTorch utiliza a biblioteca `onnx` para gerar e validar o arquivo de protocolo de grafo. No ambiente de desenvolvimento, instale com:  
> `uv add onnx` (ou `pip install onnx`).  
> Já no servidor final de deploy, apenas `onnxruntime` será necessário.

```python
from pathlib import Path
import torch
import torch.nn as nn

def exportar_para_onnx(
    modelo: nn.Module,
    caminho_saida: str | Path,
    tamanho_entrada: tuple[int, int] = (224, 224),
    opset_version: int = 17,
) -> None:
    """
    Exporta uma arquitetura PyTorch para o formato interoperável ONNX com dimensões dinâmicas.

    Args:
        modelo: Rede neural com pesos ajustados.
        caminho_saida: Destino do arquivo exportado (.onnx).
        tamanho_entrada: Resolução espacial (H, W).
        opset_version: Versão do conjunto de operadores ONNX (17+ recomendado).
    """
    modelo.eval().to("cpu")
    dummy_input = torch.randn(1, 3, *tamanho_entrada, dtype=torch.float32, device="cpu")

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model=modelo,
        args=dummy_input,
        f=str(caminho_saida),
        export_params=True,             # Serializa os pesos treinados dentro do próprio arquivo
        opset_version=opset_version,     # Versão dos operadores matemáticos
        do_constant_folding=True,        # Pré-computa constantes em tempo de compilação
        input_names=["input"],           # Nome da porta de entrada do grafo
        output_names=["logits"],         # Nome da porta de saída do grafo
        dynamic_axes={
            "input": {0: "batch_size"},   # Permite variação dinâmica no tamanho do batch
            "logits": {0: "batch_size"},  # Saída escala proporcionalmente ao batch
        },
    )
    print(f"✅ Modelo ONNX exportado com sucesso: {caminho_saida}")
```

---

### 4.3 Servindo em Produção com ONNX Runtime (Zero Dependência de PyTorch)

Em um ambiente enxuto de deploy (ex.: Docker mínimo para nuvem), não instalamos PyTorch. Apenas:

```bash
pip install onnxruntime        # Para inferência ultrarrápida em CPU
# ou
pip install onnxruntime-gpu    # Para inferência acelerada em GPU NVIDIA (CUDA / TensorRT)
```

O pipeline de inferência torna-se 100% desacoplado e opera puramente com matrizes NumPy:

```python
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

class PredictorONNX:
    """
    Servidor de inferência independente executado via ONNX Runtime.
    Não requer a instalação do framework PyTorch no ambiente de produção.
    """
    def __init__(self, caminho_onnx: str | Path, provider: str = "CPUExecutionProvider") -> None:
        self.session = ort.InferenceSession(str(caminho_onnx), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, imagem_pil: Image.Image, size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Pré-processamento idêntico ao torchvision.transforms sem importar PyTorch:
        1. Redimensionamento
        2. Normalização Z-score com estatísticas do ImageNet
        3. Transposição para formato NCHW [1, 3, H, W] em float32
        """
        img = imagem_pil.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, C] em [0.0, 1.0]

        # Estatísticas ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_np - mean) / std

        # Transpõe [H, W, C] -> [C, H, W] e adiciona dimensão de Batch [1, C, H, W]
        img_nchw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_nchw, axis=0)

    def prever(self, imagem_pil: Image.Image) -> tuple[int, float]:
        """
        Executa a inferência e retorna (índice da classe, probabilidade percentual).
        """
        tensor_np = self.preprocess(imagem_pil)
        
        # Execução direta no motor C++ do ONNX Runtime
        saidas = self.session.run([self.output_name], {self.input_name: tensor_np})
        logits = saidas[0][0]  # shape: [num_classes]

        # Softmax manual em NumPy: e^(x - max) / sum(e^(x - max))
        exp_logits = np.exp(logits - np.max(logits))
        probabilidades = exp_logits / np.sum(exp_logits)
        
        idx_vencedor = int(np.argmax(probabilidades))
        confianca = float(probabilidades[idx_vencedor] * 100.0)

        return idx_vencedor, confianca
```

---

### 4.4 Validação e Paridade Numérica

Antes de promover um modelo exportado para produção, é mandatório realizar o teste de **equivalência matemática**. O objetivo é comprovar que a discrepância entre a saída do PyTorch original e a saída do compilador otimizado é desprezível (erros de truncamento de ponto flutuante $\le 10^{-5}$).

```python
import numpy as np
import torch
import onnxruntime as ort

def validar_equivalencia_numerica(
    modelo_pytorch: torch.nn.Module,
    caminho_onnx: str,
    tolerancia_absoluta: float = 1e-5,
) -> bool:
    """
    Comprova que o modelo ONNX gera predições estritamente idênticas ao PyTorch original.
    """
    modelo_pytorch.eval()
    amostra = torch.randn(2, 3, 224, 224, dtype=torch.float32)

    # 1. Saída de referência no PyTorch
    with torch.no_grad():
        saida_pytorch = modelo_pytorch(amostra).cpu().numpy()

    # 2. Saída no ONNX Runtime
    session = ort.InferenceSession(caminho_onnx, providers=["CPUExecutionProvider"])
    nome_entrada = session.get_inputs()[0].name
    nome_saida = session.get_outputs()[0].name
    saida_onnx = session.run([nome_saida], {nome_entrada: amostra.numpy()})[0]

    # 3. Diferença absoluta máxima entre os dois grafos
    diferenca_maxima = np.max(np.abs(saida_pytorch - saida_onnx))
    print(f"📊 Diferença máxima absoluta: {diferenca_maxima:.8f}")

    # 4. Asserção matemática formal
    np.testing.assert_allclose(
        saida_pytorch,
        saida_onnx,
        rtol=1e-3,
        atol=tolerancia_absoluta,
        err_msg="Divergência numérica detectada entre PyTorch e ONNX!",
    )
    print("✅ Validação concluída: O modelo exportado possui 100% de fidelidade numérica.")
    return True
```

---

### 4.5 Matriz de Decisão de Deploy: Qual Formato Escolher?

| Critério | PyTorch Eager (`.pth`) | TorchScript (`.pt`) | ONNX (`.onnx`) |
| :--- | :--- | :--- | :--- |
| **Principal Caso de Uso** | Pesquisa e Treinamento iterativo. | Produção corporativa focada no ecossistema PyTorch/C++. | Produção multiplataforma, Microsserviços e Borda. |
| **Dependência de PyTorch** | Obrigatória (pesada, ~3 GB). | Obrigatória em Python / LibTorch em C++. | **Nenhuma**. Requer apenas `onnxruntime` (~50 MB). |
| **Requer Código Python Original?** | **Sim** (precisa importar a classe). | **Não** (grafo auto-contido). | **Não** (especificação de nós aberta). |
| **Aceleração de Hardware** | CUDA / ROCm. | CUDA / LibTorch C++ / Vulkan. | TensorRT, OpenVINO, DirectML, CoreML, CUDA. |
| **Suporte a Controle Dinâmico** | Nativo (sintaxe Python livre). | Suportado apenas com `torch.jit.script`. | Limitado (operações de grafo estático). |
| **Recomendação no AgroVision** | Ambiente de desenvolvimento. | Microsserviços baseados em C++ / TorchServe. | **API FastAPI de produção e Web/Mobile.** |

