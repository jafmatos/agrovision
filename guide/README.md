# ⚡ Trilha de Especialização: TorchVision & Refatoração do AgroVision

> **Objetivo:** Dominar a biblioteca oficial **TorchVision** através da **refatoração prática do projeto AgroVision**. Em vez de estudar conceitos isolados no vácuo, cada guia mostra o **código manual construído anteriormente**, a **dor/limitação dessa abordagem** e **como o TorchVision resolve o problema** com elegância e performance profissional.

---

## 🗺️ Mapa de Refatoração do AgroVision

| Módulo do AgroVision | Como fizemos "na unha" | A Dor / Limitação | Como o TorchVision resolve |
| :--- | :--- | :--- | :--- |
| **Modelagem**<br>([`simple_cnn.py`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/models/simple_cnn.py)) | Arquitetura CNN do zero com ~1.300 fotos de treino. | Treino lento, acurácia limitada (~80%) e risco de overfitting. | **[Guia 01](01_MODELOS_PRE_TREINADOS_E_TRANSFER_LEARNING.md):** Transfer Learning em 2 fases com ResNet-18 no [`resnet_custom.py`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/models/resnet_custom.py) (95%+ acc). |
| **Ingestão de Dados**<br>([`dataset.py`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/data/dataset.py)) | 50 linhas de código com `Path.glob()`, loops e dicionário manual. | Código boilerplate extenso, manutenção manual de classes. | **[Guia 02](02_INGESTAO_DE_DADOS_E_IO.md):** `datasets.ImageFolder` em 1 linha e I/O nativo em C++ com `torchvision.io`. |
| **Transformações**<br>([`transforms.py`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/data/transforms.py)) | PIL, `numpy.random.rand()`, divisão por 255 e `.transpose(2,0,1)`. | Preso na CPU, vai-e-volta NumPy/PIL, sem normalização ImageNet. | **[Guia 03](03_TRANSFORMS_V2_E_DATA_AUGMENTATION.md):** `transforms.v2` nativo em Tensores e normalização estatística ImageNet. |
| **Auditoria & Deploy**<br>([`predictor.py`](file:///home/jos/Desktop/Machine%20Learning/Deep%20Learning/TorchVision/src/deep_learning/engine/predictor.py)) | Inferência direta sem validação visual de onde a rede olha. | Decisão em "caixa-preta": a rede olhou a folha ou o fundo? | **[Guia 04](04_INTERPRETABILIDADE_GRADCAM_E_DEPLOY.md):** Auditoria com Grad-CAM e compilação para produção (ONNX / TorchScript). |

---

## 📚 Índice da Trilha com Foco em Refatoração

```
┌────────────────────────────────────────────────────────────────────────┐
│  ⭐ ETAPA 1: O Núcleo do Salto de Acurácia                             │
│     • Guia 01: [Modelos Pré-Treinados e Transfer Learning](01_MODELOS_PRE_TREINADOS_E_TRANSFER_LEARNING.md)
│       Substituindo a SimpleCNN pela ResNet-18 no resnet_custom.py      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  🔧 ETAPA 2: Acelerando o Pipeline de Dados                           │
│     • Guia 02: [Ingestão de Dados e Leitura com io](02_INGESTAO_DE_DADOS_E_IO.md)
│       Substituindo as 50 linhas do dataset.py pelo ImageFolder         │
│     • Guia 03: [Transforms v2 e Data Augmentation Moderno](03_TRANSFORMS_V2_E_DATA_AUGMENTATION.md)
│       Substituindo os loops de PIL do transforms.py pelo v2.Compose    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  🚀 ETAPA 3: Auditoria Visual e Produção                               │
│     • Guia 04: [Interpretabilidade com Grad-CAM e Deploy](04_INTERPRETABILIDADE_GRADCAM_E_DEPLOY.md)
│       Auditando predições e preparando o app Streamlit para produção   │
│     • Guia 05: [Guia de Bolso: TorchVision 80/20](05_GUIA_DE_BOLSO_8020.md)  
└────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Scripts Práticos Prontos para Uso

| Script | O que demonstra no contexto do AgroVision |
| :--- | :--- |
| [`exemplo_transfer_learning.py`](exemplo_transfer_learning.py) | O pipeline completo em 2 fases pronto para levar para o `resnet_custom.py`. |
| [`exemplo_transforms_v2.py`](exemplo_transforms_v2.py) | Comparação do processamento em lote com `transforms.v2` vs o manual do `transforms.py`. |
| [`exemplo_gradcam.py`](exemplo_gradcam.py) | Demonstração estritamente tipada do Grad-CAM com hooks seguros, context manager e fusão de imagem. |
| [`exemplo_export_deploy.py`](exemplo_export_deploy.py) | Pipeline completo de compilação para TorchScript, exportação ONNX (batch dinâmico) e validação numérica. |
