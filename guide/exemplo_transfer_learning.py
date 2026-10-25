"""
exemplo_transfer_learning.py
Demonstração prática dos conceitos de Transfer Learning com TorchVision:
1. Carregamento de arquitetura com pesos oficiais pré-treinados (ResNet-18)
2. Congelamento total da base convolucional (Feature Extraction)
3. Substituição da camada linear final (fc) para 3 classes
4. Verificação de parâmetros treináveis vs congelados
5. Demonstração de Fine-Tuning progressivo com taxas de aprendizado diferenciais
"""

import torch
import torch.nn as nn
from torchvision import models


def demonstrar_feature_extraction(num_classes=3):
    print("=" * 65)
    print("1. FEATURE EXTRACTION COM RESNET-18 (CONGELAMENTO TOTAL)")
    print("=" * 65)

    # 1. Carrega pesos oficiais do ImageNet
    pesos = models.ResNet18_Weights.DEFAULT
    modelo = models.resnet18(weights=pesos)
    print("✓ Pesos oficiais do ImageNet carregados com sucesso.")

    # 2. Congela todos os parâmetros da rede
    for param in modelo.parameters():
        param.requires_grad = False
    print("✓ Base convolucional congelada (requires_grad = False).")

    # 3. Lê o número de conexões de entrada da camada final
    num_features = modelo.fc.in_features
    print(f"✓ Conexões de entrada da penúltima camada: {num_features}")

    # 4. Substitui a cabeça para o nosso problema
    modelo.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )
    print(f"✓ Nova cabeça linear adaptada para {num_classes} classes.")

    # 5. Inspeciona parâmetros
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    print(f"\nTotal de parâmetros da rede:     {total_params:,}")
    print(f"Parâmetros treináveis (apenas fc): {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")

    # 6. Teste de passagem direta (Forward Pass)
    lote_teste = torch.randn(2, 3, 224, 224)  # 2 imagens aleatórias
    logits = modelo(lote_teste)
    probabilidades = torch.softmax(logits, dim=1)

    print("\n✓ Passagem direta (Forward Pass) concluída com sucesso:")
    print(f"  Shape de saída dos logits: {logits.shape}")
    print(f"  Probabilidades do primeiro exemplo: {probabilidades[0].detach().numpy().round(4)}")

    return modelo


def demonstrar_fine_tuning(modelo):
    print("\n" + "=" * 65)
    print("2. FINE-TUNING PROGRESSIVO COM LEARNING RATE SPLITTING")
    print("=" * 65)

    # 1. Destrava apenas o último bloco convolucional (layer4)
    for param in modelo.layer4.parameters():
        param.requires_grad = True

    # 2. Configura otimizador com grupos de taxas de aprendizado
    otimizador = torch.optim.Adam(
        [
            # Taxa minúscula para refino cirúrgico dos filtros do layer4:
            {"params": modelo.layer4.parameters(), "lr": 1e-5},
            # Taxa padrão na cabeça de classificação:
            {"params": modelo.fc.parameters(), "lr": 1e-3},
        ]
    )

    novos_treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"✓ layer4 destravado.")
    print(f"  Novos parâmetros treináveis: {novos_treinaveis:,}")
    print("✓ Otimizador configurado com Learning Rate Splitting:")
    print("    - layer4 lr: 0.00001 (1e-5) -> micropassos para preservar os filtros")
    print("    - fc     lr: 0.001   (1e-3) -> aprendizado normal da cabeça")


def main():
    modelo = demonstrar_feature_extraction(num_classes=3)
    demonstrar_fine_tuning(modelo)
    print("\n" + "=" * 65)
    print("✓ DEMONSTRAÇÃO CONCLUÍDA: Pipeline 100% calibrado.")
    print("=" * 65)


if __name__ == "__main__":
    main()
