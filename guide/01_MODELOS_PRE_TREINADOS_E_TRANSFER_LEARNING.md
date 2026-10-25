# 🧠 Guia 01: Modelos Pré-Treinados & Transfer Learning

> **Prioridade:** ⭐ **Máxima (O Núcleo da Visão Computacional)**
> **Objetivo:** Dominar a reutilização de modelos consagrados do **TorchVision**, dissecar a **anatomia estrutural** de uma rede pré-treinada, entender as estratégias de **Feature Extraction** e **Fine-Tuning Progressivo**, aplicar taxas de aprendizado diferenciais (*Learning Rate Splitting*) e adaptar diferentes famílias de cabeças lineares (`fc` vs `classifier`).

---

## 1. Por que Modelos Pré-Treinados são o Padrão da Indústria?

Treinar uma Convolutional Neural Network (CNN) profunda do zero exige:

1. **Dados Massivos:** Centenas de milhares de fotos anotadas por humanos;
2. **Tempo & Custo:** Semanas de treinamento em dezenas de GPUs corporativas;
3. **Risco de Overfitting:** Redes profundas com milhões de parâmetros decoram datasets pequenos facilmente.

No mundo real, **mais de 95% das soluções profissionais aplicam Transfer Learning**: partimos de uma rede que já foi treinada por dias na base **ImageNet-1k** (1,28 milhão de imagens distribuídas em 1.000 categorias de objetos).

### Como a rede neural hierarquiza o mundo visual:

Uma rede pré-treinada no ImageNet desenvolveu uma hierarquia visual sofisticada:

```
[ Imagem de Entrada: 3 x 224 x 224 ]
                 │
                 ▼
     [ Camadas Iniciais (ex: conv1) ]
     ──► Detectores de bordas, linhas verticais/horizontais, gradientes e contrastes.
     ──► UNIVERSAIS: São exatamente os mesmos para folhas, carros, prédios ou tecidos médicos.
                 │
                 ▼
     [ Camadas Intermediárias (ex: layer2, layer3) ]
     ──► Combinações de bordas formando texturas, cantos, rugosidades, malhas e pontilhados.
                 │
                 ▼
     [ Camadas Profundas (ex: layer4) ]
     ──► Formas complexas e geometrias de alto nível do ImageNet (olhos, asas, rodas).
                 │
                 ▼
     [ Classificador Linear Final (fc / classifier) ]
     ──► Camada estritamente linear que distribui pesos para as 1.000 classes originais.
```

Para adaptar essa rede para o nosso problema (por exemplo, 3 classes de folhas: *Saudável*, *Oídio* e *Ferrugem*), **reaproveitamos todo o corpo extrator de características** e apenas substituímos a camada de classificação final.

---

### 🌿 O Caso AgroVision: A Dor da `SimpleCNN` vs A Solução no `resnet_custom.py`

No projeto AgroVision, construímos a [`SimpleCNN`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/models/simple_cnn.py>) do zero:

```python
# Trecho do seu src/deep_learning/models/simple_cnn.py
self.features = nn.Sequential(...)  # 3 blocos Conv2d + BatchNorm + MaxPool
self.gap = nn.AdaptiveAvgPool2d((1, 1))
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(64, num_classes)
)
```

#### Qual é o problema prático dessa abordagem?

1. **Poucos Dados:** O nosso dataset tem apenas **1.322 fotos de treino** divididas entre as 3 classes (*Saudável*, *Oídio*, *Ferrugem*).
2. **Começar do Zero:** A `SimpleCNN` começa com pesos aleatórios. Ela precisa de dezenas de épocas apenas para aprender o básico (o que é uma borda, um gradiente, uma curva).
3. **Teto de Performance:** O modelo atinge um platô de ~80–85% de acurácia e tem forte tendência a sofrer *overfitting* (decorar a cor do solo ou o ângulo da foto em vez dos sintomas do fungo).

#### A Solução com TorchVision:

No arquivo em branco [`src/deep_learning/models/resnet_custom.py`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/models/resnet_custom.py>), vamos substituir o treino do zero por uma **ResNet-18 pré-treinada**, aproveitando os filtros do ImageNet para saltar de 80% para **95%+** em poucas épocas!

---

## 2. A Estrutura Universal de Qualquer Modelo Pré-Treinado

Independentemente da arquitetura escolhida — seja **ResNet**, **MobileNet**, **EfficientNet**, **ConvNeXt** ou até a sua própria [`SimpleCNN`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/models/simple_cnn.py>) —, praticamente toda rede neural moderna de Visão Computacional é construída sobre o **mesmo esqueleto de 3 partes**:

```
[ Imagem de Entrada: (B, 3, H, W) ]
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. STEM (Porta de Entrada)                                             │
│    Recebe os pixels RGB brutos e faz a primeira redução de resolução,  │
│    projetando as cores para os primeiros canais da rede.               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. BACKBONE / CORPO (Extrator de Características)                      │
│    O "cérebro visual" da rede. Uma sequência de blocos onde:           │
│    • A resolução espacial (H, W) diminui progressivamente;             │
│    • O número de canais (C) aumenta, acumulando riqueza de filtros;    │
│    • A hierarquia vai de bordas/texturas simples até formas complexas. │
│    👉 NO TRANSFER LEARNING: É a parte que REAPROVEITAMOS e CONGELAMOS. │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. PONTE & CABEÇA (Global Pooling & Classification Head)               │
│    • Pooling Global (GAP): Colapsa o mapa 2D em um vetor unidimensional│
│    • Cabeça Linear (Head): Conecta os atributos aos logits das classes.│
│    👉 NO TRANSFER LEARNING: É a ÚNICA parte que ARRANCAMOS e TROCAMOS. │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
[ Saída: Logits das Classes (B, num_classes) ]
```

---

### O Papel de Cada Parte no Ciclo Visual:

1. **Stem (Porta de Entrada):**

   * *O que faz:* Pega a imagem crua de alta resolução e aplica uma primeira convolução e redução de escala.
   * *Por que existe:* Processar imagens em alta resolução em muitas camadas é caro. O Stem reduz o tamanho espacial rapidamente para que o resto da rede possa se concentrar em combinar filtros e canais.
2. **Backbone / Corpo (O Extrator de Atributos):**

   * *O que faz:* É onde reside mais de 90% dos parâmetros e do aprendizado da rede. É estruturado em múltiplos estágios que seguem uma **regra universal**:
     $$
     \text{A rede se aprofunda} \implies \text{A resolução (H, W) diminui} \quad \text{e} \quad \text{Os canais (C) aumentam}
     $$
   * *A hierarquia universal:*
     * **Início:** Detectores de linhas, bordas, contrastes e cantos (universais para qualquer imagem do mundo).
     * **Meio:** Combinações de bordas formando texturas, rugosidades e malhas.
     * **Fim:** Geometrias ricas e peças completas dos objetos do ImageNet.
3. **Ponte & Cabeça (Global Pooling e Classificador):**

   * **Global Average Pooling (GAP):** Em vez de ligar milhares de conexões pesadas, quase toda rede moderna tira uma média simples de cada canal no final, transformando o mapa bidimensional em um vetor de tamanho fixo (ex: 512 ou 1.280 números).
   * **A Cabeça (Head):** É simplesmente uma ou duas camadas lineares que recebem esse vetor e dizem: *"com base nessas características, essa imagem é a classe A, B ou C"*.

---

### Como diferentes modelos nomeiam essas partes no PyTorch:

Todas as redes seguem essa mesma divisão lógica; apenas os nomes dos atributos no código mudam ligeiramente:

| Família do Modelo            | Corpo Extrator (Backbone)                  | Ponte de Redução | Cabeça de Classificação (Head) |
| :---------------------------- | :----------------------------------------- | :----------------- | :-------------------------------- |
| **ResNet**              | `modelo.conv1` + `layer1` a `layer4` | `modelo.avgpool` | `modelo.fc`                     |
| **MobileNet**           | `modelo.features`                        | `modelo.avgpool` | `modelo.classifier`             |
| **EfficientNet**        | `modelo.features`                        | `modelo.avgpool` | `modelo.classifier`             |
| **A sua `SimpleCNN`** | `self.features`                          | `self.gap`       | `self.classifier`               |

Perceba a elegância: não importa a rede que você escolher na literatura, **o princípio do Transfer Learning é sempre idêntico**: você mantém o **Corpo** (congelado) e substitui a **Cabeça** para o seu número de classes!

---

## 3. O Pipeline Canônico em 2 Fases (O Treinamento Ótimo)

Muitos tutoriais introdutórios apresentam "Feature Extraction" e "Fine-Tuning" como se fossem duas opções separadas em um cardápio (ou você escolhe uma, ou escolhe a outra). **Na prática profissional, isso é uma falsa dicotomia.**

A estratégia que entrega o **desempenho ótimo** em qualquer problema real de Visão Computacional une as duas em um **pipeline contínuo de dois tempos**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 1: AQUECIMENTO DA CABEÇA (Head Warm-up)                           │
│ • Backbone: 100% CONGELADO (requires_grad = False)                     │
│ • Cabeça:   TREINANDO com taxa normal (lr = 1e-3)                      │
│ • Duração:  5 a 10 épocas                                              │
│ 👉 Objetivo: Tirar a cabeça do ruído aleatório e estabilizar a perda.  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ (Cabeça calibrada)
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 2: REFINO CIRÚRGICO (Fine-Tuning com Learning Rate Splitting)     │
│ • Base inicial:  CONTINUA CONGELADA (filtros universais protegidos)    │
│ • Último bloco:  DESTRAVADO com micropassos (lr = 1e-5)                │
│ • Cabeça:        TREINANDO com taxa padrão (lr = 1e-3 ou 1e-4)         │
│ • Duração:       mais 5 a 10 épocas                                    │
│ 👉 Objetivo: Refinar texturas para o seu domínio e atingir 95%+ acc.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Implementação Prática do Fluxo Completo:

```python
import torch
import torch.nn as nn
from torchvision import models

# =====================================================================
# ETAPA 1: PREPARAÇÃO DO MODELO E FASE 1 (AQUECIMENTO)
# =====================================================================

# 1. Carrega o modelo com os pesos oficiais do ImageNet
pesos = models.ResNet18_Weights.DEFAULT
modelo = models.resnet18(weights=pesos)

# 2. Congela o modelo inteiro (escudo protetor contra a cabeça aleatória)
for param in modelo.parameters():
    param.requires_grad = False

# 3. Substitui a cabeça para o seu número de classes (ex: 3)
num_features = modelo.fc.in_features
modelo.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(128, 3)
)

# 4. Otimizador da Fase 1: Atualiza APENAS a cabeça
otimizador_fase1 = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)

# -> Execute aqui o loop de treino por 5 a 10 épocas com otimizador_fase1!
# -> A cabeça aprende a ler os atributos sem abalar nenhum filtro prévio.

# =====================================================================
# ETAPA 2: TRANSIÇÃO PARA A FASE 2 (REFINO CIRÚRGICO)
# =====================================================================

# 5. Destrava apenas o último bloco convolucional (layer4 na ResNet ou features[-1] na MobileNet)
for param in modelo.layer4.parameters():
    param.requires_grad = True

# 6. Otimizador da Fase 2 com Learning Rate Splitting:
#    - 1e-5 no layer4 (passo 100x menor para não descalibrar os filtros)
#    - 1e-3 na cabeça (ajuste responsivo contínuo)
otimizador_fase2 = torch.optim.Adam([
    {'params': modelo.layer4.parameters(), 'lr': 1e-5},
    {'params': modelo.fc.parameters(),     'lr': 1e-3}
])

# -> Execute aqui mais 5 a 10 épocas com otimizador_fase2!
# -> O modelo atinge seu ponto ótimo de acurácia e generalização.
```

### Por que esse fluxo em dois tempos é o padrão da indústria?

* Se você tentar ir direto para o Fine-Tuning desde a época 1, os gradientes caóticos da cabeça aleatória degradam o backbone (*Catastrophic Forgetting*).
* Se você parar na Fase 1 e nunca fizer o Fine-Tuning, a rede não especializa suas texturas finais para as patologias do seu domínio, deixando de 5% a 15% de acurácia na mesa.
* Ao juntar os dois passos na sequência correta, você atinge a **máxima convergência com risco zero**.

---

## 4. O Zoológico de Arquiteturas: Como Escolher?

O TorchVision disponibiliza mais de 60 arquiteturas pré-treinadas. Três famílias cobrem 99% das necessidades práticas:

```
                  ┌───────────────────────────────────────────────┐
                  │ Qual é o seu objetivo de implementação?       │
                  └───────┬───────────────────────────────┬───────┘
                          │                               │
                [ Servidor / Desktop ]            [ Mobile / Borda / IoT ]
                          │                               │
                          ▼                               ▼
                   Família ResNet                MobileNetV3-Small
                 (ResNet-18 / 50)               (~10 MB / 2.5M params)
```

| Modelo                           | Tamanho (Disco)  | Parâmetros     | Top-1 Acc (ImageNet) | Melhor Cenário                                        |
| :------------------------------- | :--------------- | :-------------- | :------------------- | :----------------------------------------------------- |
| **`mobilenet_v3_small`** | **~10 MB** | **2.5 M** | 67.7%                | Celulares, Raspberry Pi, drones, baixa latência       |
| **`mobilenet_v3_large`** | ~22 MB           | 5.4 M           | 75.3%                | Dispositivos móveis modernos com maior acurácia      |
| **`resnet18`**           | ~45 MB           | 11.7 M          | 69.8%                | O modelo padrão para prototipagem rápida e estável  |
| **`resnet50`**           | ~100 MB          | 25.6 M          | 80.9%                | Produção corporativa, servidores e benchmarks        |
| **`convnext_tiny`**      | ~114 MB          | 28.6 M          | 82.1%                | Arquitetura convolucional moderna de última geração |

---

## 5. Como adaptar a Cabeça em Diferentes Modelos: `fc` vs `classifier`

Cada família de modelo no TorchVision organiza sua camada final de forma ligeiramente diferente. É fundamental saber inspecionar e substituir corretamente:

### Caso 1: Família ResNet (`modelo.fc`)

Nas ResNets, a cabeça é um atributo simples chamado `.fc`:

```python
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
in_features = modelo.fc.in_features  # 2048 na ResNet-50
modelo.fc = nn.Linear(in_features, num_classes)
```

### Caso 2: Família MobileNetV3 (`modelo.classifier`)

Na MobileNet, a cabeça é um contêiner `nn.Sequential` chamado `.classifier`. O classificador final é o elemento no **índice 3**:

```python
modelo = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Inspeciona o número de entradas do último linear (índice 3)
in_features = modelo.classifier[3].in_features  # 1024

# Substitui apenas a camada final preservando o dropout e o pooling anterior
modelo.classifier[3] = nn.Linear(in_features, num_classes)
```

### Caso 3: Família EfficientNet (`modelo.classifier`)

Na EfficientNet, a cabeça também se chama `.classifier`, mas a camada linear de saída é o **índice 1**:

```python
modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = modelo.classifier[1].in_features  # 1280
modelo.classifier[1] = nn.Linear(in_features, num_classes)
```

---

## 6. O Atalho Oficial: `weights.transforms()`

Historicamente, muitos desenvolvedores erravam métricas de Transfer Learning por aplicar transformações incompatíveis com o que o modelo esperava (por exemplo, tamanho errado ou normalização incorreta).

No TorchVision moderno, **os pesos oficiais encapsulam o pipeline exato de transformações**:

```python
pesos = models.ResNet18_Weights.DEFAULT

# Gera instantaneamente o transform calibrado:
transform_oficial = pesos.transforms()

print(transform_oficial)
# Saída:
# ImageClassification(
#     crop_size=[224],
#     resize_size=[256],
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225],
#     interpolation=InterpolationMode.BILINEAR
# )
```

Você pode passar esse objeto diretamente para o seu `Dataset` de validação ou de teste:

```python
dataset_teste = datasets.ImageFolder(root="data/test", transform=pesos.transforms())
```

Isso garante 100% de precisão numérica sem precisar consultar manuais externos.
