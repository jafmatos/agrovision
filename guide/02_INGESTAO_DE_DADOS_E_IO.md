# 🚀 Guia 02: Ingestão de Dados e Leitura com `torchvision.io`

> **Prioridade:** 🔧 **Alta (Utilitário de Alto Desempenho)**
> **Objetivo:** Automatizar a leitura de diretórios com **`datasets.ImageFolder`**, acelerar a ingestão de imagens com **`torchvision.io`** nativo em C++ e configurar pipelines de dados no **`DataLoader`** para eliminar gargalos de GPU.

---

## 1. Carregamento Automático com `datasets.ImageFolder`

### 🌿 O Caso AgroVision: A Dor do `dataset.py`

No arquivo [`src/deep_learning/data/dataset.py`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/data/dataset.py>), tivemos que escrever mais de 50 linhas de código manual para herdar de `Dataset`:

```python
# O código manual que construímos em src/deep_learning/data/dataset.py
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path = DATA_DIR, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Varredura manual de pastas e classes
        self.classes = sorted([dir.name for dir in self.root_dir.iterdir() if dir.is_dir()])
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        # Loop manual para filtrar extensões válidas (.jpg, .png)
        for cls_name in self.classes:
            folder = self.root_dir / cls_name
            for img_path in folder.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                    self.samples.append((img_path, self.class_to_index[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

#### Qual era o problema dessa abordagem?

1. **Boilerplate Excessivo:** Mais de 50 linhas de código que todo desenvolvedor precisa repetir em todo projeto.
2. **Fragilidade:** Tratamento manual de caminhos, barras de diretório (`/`) e extensões de imagem.
3. **Falta de Otimização:** O `Image.open` do PIL não aproveita otimizações em nível de sistema operacional e C++.

---

### A Solução TorchVision: 1 Linha que Elimina o `dataset.py`

O TorchVision já vem de fábrica com a classe [`datasets.ImageFolder`](<file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/data/dataset.py#L21-L40>), que substitui **todo o seu arquivo `dataset.py`** por uma única chamada:

### Estrutura de Diretórios Esperada:

```text
data/
├── train/
│   ├── ferrugem/             <-- todas as imagens da classe 'ferrugem'
│   ├── oidio/                <-- todas as imagens da classe 'oidio'
│   └── saudavel/             <-- todas as imagens da classe 'saudavel'
└── test/
    ├── ferrugem/
    ├── oidio/
    └── saudavel/
```

### Código de Instanciação:

```python
from torchvision import datasets
from torchvision.transforms import v2

# 1. Criação do Dataset mapeando as pastas de treino
dataset_treino = datasets.ImageFolder(
    root="dados/train",
    transform=transform_treino  # Pipeline de transformações (visto no Guia 03)
)

# 2. Inspeciona a lista ordenada de classes detectadas
print("Classes encontradas:", dataset_treino.classes)
# Saída: ['ferrugem', 'oidio', 'saudavel']

# 3. Mapeamento de rótulos de texto para IDs inteiros
print("Mapeamento class_to_idx:", dataset_treino.class_to_idx)
# Saída: {'ferrugem': 0, 'oidio': 1, 'saudavel': 2}

# 4. Inspeciona a quantidade total de amostras indexadas
print("Total de amostras:", len(dataset_treino))
```

---

## 2. Leitura Nativa e Ultra-Rápida com `torchvision.io`

Ao processar milhares de imagens por segundo, abrir arquivos usando PIL ou OpenCV cria um overhead de conversão contínua entre Python, buffers de memória e arrays NumPy.

O módulo **`torchvision.io`** implementa decodificadores JPEG/PNG diretamente em **C++**, carregando a imagem do disco direto para um **Tensor PyTorch**:

```python
from torchvision.io import read_image, ImageReadMode

# Lê a imagem diretamente para um Tensor [C, H, W] uint8
tensor_imagem = read_image("amostra_folha.jpg", mode=ImageReadMode.RGB)

print("Formato do Tensor:", tensor_imagem.shape)  # torch.Size([3, 1080, 1920])
print("Tipo numérico:", tensor_imagem.dtype)       # torch.uint8
print("Faixa de valores:", tensor_imagem.min().item(), "a", tensor_imagem.max().item())  # 0 a 255
```

> [!WARNING]
> **Atenção em Ambientes AMD ROCm ou Builds Sem C++ I/O**:
> Se ao executar `read_image` você receber o erro `RuntimeError: decode_jpeg: torchvision not compiled with libjpeg support`, significa que o binário do TorchVision da sua plataforma (comum em wheels ROCm da AMD) foi compilado sem a `libjpeg-turbo`.
> 
> Nesses ambientes, a alternativa oficial recomendada pelo PyTorch é usar o decodificador do Pillow integrado ao **Transforms v2**:
> ```python
> from PIL import Image
> from torchvision.transforms import v2
> 
> # Carrega com PIL e converte diretamente para Tensor [C, H, W] uint8:
> tensor_imagem = v2.functional.to_image(Image.open("amostra_folha.jpg").convert("RGB"))
> ```
> O resultado em memória é 100% equivalente ao `read_image`. É exatamente por isso que o `datasets.ImageFolder` usa o Pillow por padrão, garantindo portabilidade entre CUDA, ROCm, Mac e CPU.

### Salvando Tensores de volta no disco:

```python
from torchvision.io import write_jpeg

# Salva um tensor diretamente como arquivo JPEG comprimido
write_jpeg(tensor_imagem, "saida_processada.jpg", quality=90)
```

---

## 3. Otimização do `DataLoader` para Eliminar Gargalos de Hardware

Um dos problemas mais comuns em Deep Learning é a **subutilização da GPU**: a placa de vídeo passa metade do tempo ociosa esperando a CPU ler e decodificar as fotos do disco (*I/O Bottleneck*).

Para garantir que a GPU trabalhe a 100% de capacidade:

```python
import os
from torch.utils.data import DataLoader

loader_treino = DataLoader(
    dataset=dataset_treino,
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count() or 4,  # Paraleliza a leitura em múltiplos processos de CPU
    pin_memory=True,                  # Trava as páginas de memória RAM para aceleração PCIe
    persistent_workers=True,          # Mantém os processos de leitura vivos entre épocas
    prefetch_factor=2                 # Cada worker pré-carrega 2 lotes adiantados
)
```

### O que cada parâmetro resolve no hardware:

1. **`num_workers > 0`**: Cria processos paralelos na CPU para decodificar e transformar as imagens dos próximos lotes enquanto a GPU está ocupada calculando o `forward` e o `backward` do lote atual.
2. **`pin_memory=True`**: Aloca os tensores em **memória RAM travada (page-locked)**. Isso permite que a controladora da GPU copie os dados diretamente via **DMA (Direct Memory Access)** através do barramento PCIe sem passar pela CPU, acelerando a transferência em até 2x.
3. **`persistent_workers=True`**: Evita o custo computacional de destruir e recriar os processos de leitura no final de cada época do ciclo de treino.

---

## 4. Datasets Prontos do Ecossistema

Para prototipagem rápida de novos modelos ou testes comparativos sem precisar baixar ou organizar pastas manualmente:

```python
from torchvision.datasets import CIFAR10

# Baixa e instancia o dataset em um único comando
cifar = CIFAR10(
    root="./data", 
    train=True, 
    download=True, 
    transform=transform_treino
)
```
