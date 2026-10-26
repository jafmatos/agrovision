# 📋 Guia de Bolso: TorchVision 80/20

> **Objetivo:** O resumo de consulta rápida mais objetivo para manter aberto ao lado do seu editor de código.

---

## 1. O Núcleo: Modelos Pré-Treinados & Transfer Learning

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Carregar rede com pesos oficiais** | `modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)` | Carrega filtros refinados em 1,2 milhão de imagens do ImageNet. |
| **Pegar o transform oficial da rede** | `transform = weights.transforms()` | Garante tamanho e normalização exatos esperados pelo modelo. |
| **Feature Extraction (Congelar tudo)** | `for p in modelo.parameters(): p.requires_grad = False` | Desativa cálculo de gradientes; economiza VRAM e acelera o treino. |
| **Adaptar cabeça na ResNet** | `modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)` | Redefine a saída final para o seu número específico de classes. |
| **Adaptar cabeça na MobileNetV3** | `modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, num_classes)` | Substitui a última camada preservando o dropout anterior. |
| **Fine-Tuning com Taxas Diferenciais** | `optim.Adam([{'params': modelo.layer4.parameters(), 'lr': 1e-5}, {'params': modelo.fc.parameters(), 'lr': 1e-3}])` | Refina texturas profundas sem destruir os filtros universais prévios. |

---

## 2. Utilitários: Data Augmentation (`transforms.v2`)

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Simular ângulos variados** | `v2.RandomRotation(degrees=20)` | Evita memorização de orientações rígidas no treino. |
| **Simular fotos de perto e de longe** | `v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0))` | Ensina invariância de escala cortando e redimensionando. |
| **Simular diferentes iluminações** | `v2.ColorJitter(brightness=0.25, contrast=0.25)` | Combate overfitting a sombras e tons de luz específicos. |
| **Converter e normalizar [0, 1]** | `v2.ToDtype(torch.float32, scale=True)` | Substitui a divisão manual por 255 em tensores. |
| **Normalização padrão ImageNet** | `v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | Alinha a distribuição estatística com os pesos pré-treinados. |
| **Avaliação sem distorção** | `v2.Compose([v2.Resize(256), v2.CenterCrop(224)])` | Recorta o centro da imagem preservando a proporção de aspecto real. |

---

## 3. Utilitários: Ingestão de Dados & I/O

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Ler pastas por classes** | `dataset = datasets.ImageFolder(root="dados/treino", transform=t)` | Lê diretórios e cria mapeamento de classes automaticamente. |
| **Leitura rápida direta do disco** | `tensor = torchvision.io.read_image("foto.jpg")` | Decodifica JPEG/PNG direto em Tensor uint8 via C++. |
| **Acelerar transferência para GPU** | `DataLoader(..., pin_memory=True, num_workers=4)` | Usa DMA via barramento PCIe e pré-carregamento paralelo na CPU. |

---

## 4. Auditoria & Inferência em Produção

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Travar BatchNorm e Dropout** | `modelo.eval()` | Garante respostas determinísticas na hora da avaliação. |
| **Inferência sem gravar gradientes** | `with torch.no_grad(): logits = modelo(batch)` | Economiza memória de vídeo e tempo de processamento. |
| **Probabilidades percentuais** | `probs = torch.softmax(logits, dim=1)` | Converte valores brutos em probabilidades que somam 1.0. |
| **Auditoria Visual (Grad-CAM)** | `with GradCAM(model, layer) as gc: cam = gc.generate_heatmap(x)` | Gera mapa de calor 2D das regiões que ditaram a decisão. |
| **Compilar para TorchScript (C++)**| `torch.jit.trace(modelo, dummy).save("modelo.pt")` | Exporta grafo estático serializado sem dependência da classe Python. |
| **Exportar para ONNX (Universal)** | `torch.onnx.export(m, d, "m.onnx", dynamic_axes={"in": {0: "b"}}, opset_version=17)` | Exporta para padrão aberto com suporte a batch dinâmico. |
| **Inferência sem PyTorch (ONNX)** | `ort.InferenceSession("m.onnx").run(["logits"], {"in": x_np})` | Executa inferência em C++ usando apenas NumPy (~50 MB runtime). |
