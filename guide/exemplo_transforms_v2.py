"""
exemplo_transforms_v2.py
Demonstração prática dos utilitários de transformação com torchvision.transforms.v2:
1. Pipeline de Treino estocástico (RandomResizedCrop, Flip, Rotation, ColorJitter, Normalize)
2. Pipeline de Teste determinístico (Resize 256, CenterCrop 224, Normalize)
3. Uso do atalho oficial weights.transforms()
4. Verificação de dimensões, tipos (dtype) e faixas estatísticas (mínimo, máximo, média)
"""

import torch
from torchvision import models
from torchvision.transforms import v2


def criar_pipeline_treino():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),  # Escala [0, 255] -> [0.0, 1.0]
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def criar_pipeline_teste():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=256, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    print("=" * 65)
    print("DEMONSTRAÇÃO DE TRANSFORMS V2 COM TORCHVISION")
    print("=" * 65)

    # Simula um lote bruto de 2 imagens no formato uint8 [B, C, H, W] como lido do disco
    lote_bruto = torch.randint(0, 256, (2, 3, 300, 400), dtype=torch.uint8)
    print(f"\n1. Lote Bruto de Entrada: shape={lote_bruto.shape}, dtype={lote_bruto.dtype}")

    # 1. Aplica pipeline de treino
    pipe_treino = criar_pipeline_treino()
    lote_treino = pipe_treino(lote_bruto)
    print("\n2. Saída do Pipeline de Treino (Estocástico):")
    print(f"   Shape: {lote_treino.shape}")
    print(f"   Tipo (dtype): {lote_treino.dtype}")
    print(f"   Média aproximada: {lote_treino.mean().item():.3f}")
    print(f"   Desvio padrão aproximado: {lote_treino.std().item():.3f}")

    # 2. Aplica pipeline de teste
    pipe_teste = criar_pipeline_teste()
    lote_teste = pipe_teste(lote_bruto)
    print("\n3. Saída do Pipeline de Teste (Determinístico):")
    print(f"   Shape: {lote_teste.shape}")
    print(f"   Tipo (dtype): {lote_teste.dtype}")

    # 3. Demonstra o atalho dos pesos oficiais
    print("\n4. Atalho com weights.transforms():")
    pesos = models.ResNet18_Weights.DEFAULT
    transform_oficial = pesos.transforms()
    print(f"   Transform oficial da ResNet-18:\n   {transform_oficial}")

    print("\n" + "=" * 65)
    print("✓ DEMONSTRAÇÃO DO TRANSFORMS V2 CONCLUÍDA COM SUCESSO.")
    print("=" * 65)


if __name__ == "__main__":
    main()
