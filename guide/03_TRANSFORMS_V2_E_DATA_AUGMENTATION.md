# 🎨 Guia 03: Transforms v2 & Data Augmentation Moderno

> **Prioridade:** 🔧 **Alta (Utilitário de Suporte aos Modelos)**
> **Objetivo:** Dominar a nova API **`torchvision.transforms.v2`**, entender as estratégias de aumento de dados estocástico para treino, construir pipelines determinísticos para teste e validar por que a normalização do ImageNet é obrigatória.

---

## 1. Por que Data Augmentation é Obrigatório?

Filtros convolucionais aprendem correlações espaciais diretas na grade de pixels. No entanto, eles **não possuem invariância natural a rotação, escala ou variações fotométricas de iluminação**.

Se treinarmos uma rede profunda com um conjunto de dados estático, ela sofrerá **Overfitting** rapidamente, decorando traços específicos do fundo ou o ângulo exato em que a foto foi tirada.

O **Data Augmentation** resolve isso aplicando transformações aleatórias (estocásticas) na memória a cada época:

* A foto original sofre pequenos cortes, giros, espelhamentos e ajustes de cor;
* A rede **nunca enxerga a mesma imagem duas vezes** durante o treinamento;
* A rede é forçada a aprender atributos biológicos robustos e invariantes.

---

### 🌿 O Caso AgroVision: A Dor do `transforms.py`

No arquivo [`src/deep_learning/data/transforms.py`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/data/transforms.py>), construímos as transformações de forma manual combinando NumPy e Pillow:

```python
# O código manual que construímos em src/deep_learning/data/transforms.py
def train_transform(image: Image.Image) -> torch.Tensor:
    if numpy.random.rand() < 0.5:
        angle = float(numpy.random.uniform(-15.0, 15.0))
        image = image.rotate(angle=angle)

    if numpy.random.rand() < 0.5:
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT)

    image = ImageOps.fit(image, size=IMAGE_SIZE, method=Image.Resampling.BILINEAR)

    # Conversão manual de tipo e transposição de eixos (H, W, C) -> (C, H, W)
    tensor_numpy = numpy.array(image, dtype=numpy.float32) / 255
    tensor_numpy = tensor_numpy.transpose(2, 0, 1)

    return torch.from_numpy(tensor_numpy)
```

#### Quais problemas essa abordagem artesanal causa?

1. **Gargalo de CPU (Vai-e-volta constante):** Para cada imagem, o código faz: `PIL Image` $\to$ `rotate` $\to$ `ImageOps.fit` $\to$ `numpy.array` $\to$ transposição de eixos $\to$ `torch.from_numpy`. Essa ponte contínua de memória RAM atrasa o `DataLoader`.
2. **Incompatível com GPU em lote:** O código só consegue rodar na CPU, uma foto de cada vez.
3. **Falta de Normalização Z-Score:** O código apenas dividia por 255 (mantendo os dados entre `[0.0, 1.0]`), mas não aplicava a centralização em zero (`Normalize`) exigida para que os filtros das redes pré-treinadas funcionem com máxima acurácia.

---

## 2. A Nova Geração: `torchvision.transforms.v2`

Tradicionalmente, o TorchVision usava `torchvision.transforms`, que operava em objetos `PIL.Image` na CPU. A nova API **`v2`** revolucionou o processamento:

```
[ PIL Image / File ] ──► [ v2.ToImage() ] ──► [ Tensor PyTorch ] ──► [ GPU (Opcional) ]
                                                        │
                                                        ▼
                        Transforms v2 aplicados em Lote (Imagens, Caixas e Máscaras)
```

### Vantagens do `transforms.v2`:

1. **Executa diretamente em Tensores:** Elimina conversões desnecessárias entre PIL e NumPy.
2. **Suporte a Tarefas Múltiplas:** Aplica a mesma rotação ou corte simultaneamente na imagem, nas caixas delimitadoras (*Bounding Boxes*) e nas máscaras de segmentação.
3. **Aceleração em GPU:** Pode ser executado em lotes inteiros diretamente na VRAM.

---

## 3. O Pipeline de Treino (Estocástico e Aleatório)

```python
import torch
from torchvision.transforms import v2

transform_treino = v2.Compose([
    # 1. Garante que a entrada seja convertida em Tensor nativo
    v2.ToImage(),
  
    # 2. Recorte com zoom aleatório e redimensionamento para 224x224
    v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), antialias=True),
  
    # 3. Espelhamento horizontal com 50% de probabilidade
    v2.RandomHorizontalFlip(p=0.5),
  
    # 4. Rotação suave no plano da foto (entre -20 e +20 graus)
    v2.RandomRotation(degrees=20),
  
    # 5. Variações fotométricas de luz e cor
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
  
    # 6. Converte uint8 [0, 255] para float32 [0.0, 1.0] escalonado
    v2.ToDtype(torch.float32, scale=True),
  
    # 7. Normalização exata com as métricas do ImageNet
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

### O que cada método executa nos dados:

* **`v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0))`**:
  Recorta uma porção aleatória entre 70% e 100% da foto e reescala para $224 \times 224$. Simula a câmera fotografando tanto em enquadramento fechado (*macro*) quanto à distância.
* **`v2.RandomHorizontalFlip(p=0.5)`**:
  Inverte horizontalmente a foto. Fundamental para biologia: uma pústula de ferrugem não tem lado esquerdo ou direito preferencial.
* **`v2.ColorJitter(...)`**:
  Altera aleatoriamente o brilho ($\pm 25\%$), o contraste ($\pm 25\%$) e a vivacidade das cores ($\pm 20\%$). Simula fotos tiradas sob sol forte, dia chuvoso ou com sensores de celular com balanço de branco diferente.
* **`v2.ToDtype(torch.float32, scale=True)`**:
  O grande trunfo do `v2`: converte o tensor para ponto flutuante `float32` e, com `scale=True`, divide automaticamente por 255, normalizando o intervalo numérico para $[0.0, 1.0]$.
* **`v2.Normalize(mean, std)`**:
  Centraliza os dados em torno de zero e escala a variância para 1 em cada canal.

---

## 4. O Pipeline de Teste e Validação (Estritamente Determinístico)

> ⚠️ **Regra Fundamental de Avaliação:** **NUNCA** aplique rotações, espelhamentos ou cortes aleatórios na validação ou no teste. A avaliação deve ser estritamente reproduzível e inspecionar a imagem íntegra.

```python
transform_teste = v2.Compose([
    v2.ToImage(),
  
    # 1. Redimensiona proporcionalmente o menor lado para 256 pixels
    v2.Resize(size=256, antialias=True),
  
    # 2. Recorta os 224x224 pixels centrais exatos
    v2.CenterCrop(size=(224, 224)),
  
    # 3. Converte para float32 escalado [0.0, 1.0]
    v2.ToDtype(torch.float32, scale=True),
  
    # 4. Mesma normalização do ImageNet
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

### Por que `Resize(256)` seguido de `CenterCrop(224)`?

Se redimensionássemos uma foto retangular (ex.: $1920 \times 1080$) diretamente para uma grade quadrada de $224 \times 224$, os objetos ficariam espremidos (*distorção de aspecto*).

A convenção clássica da visão computacional é:

1. Reescalar a foto proporcionalmente até que o menor eixo atinja 256 pixels;
2. Recortar os $224 \times 224$ pixels centrais, onde o objeto de interesse naturalmente se encontra.

---

## 5. Por que a Normalização do ImageNet é Obrigatória?

A fórmula matemática aplicada pelo `Normalize` em cada pixel $x$ do canal $c$ é:

$$
x_{\text{norm}} = \frac{x - \text{mean}[c]}{\text{std}[c]}
$$

Os valores universais do TorchVision são:

* `mean = [0.485, 0.456, 0.406]` (Médias dos canais R, G e B)
* `std  = [0.229, 0.224, 0.225]` (Desvios padrão dos canais R, G e B)

### O motivo técnico:

As redes pré-treinadas (`ResNet`, `MobileNet`, etc.) foram treinadas por semanas recebendo tensores centralizados nessa exata distribuição estatística. Se você alimentar o modelo com valores brutos $[0, 1]$ ou $[0, 255]$, a distribuição dos dados de entrada mudará drasticamente (*covariate shift*), fazendo com que as ativações das camadas convolucionais fiquem saturadas ou ineficazes.
