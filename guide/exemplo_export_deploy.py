"""
exemplo_export_deploy.py
Demonstração prática e estritamente tipada de exportação para produção no TorchVision:
1. Compilação para TorchScript via Tracing (torch.jit.trace)
2. Salvamento e carregamento de modelo compilado (.pt) sem a classe Python
3. Exportação para formato universal ONNX com eixos dinâmicos (batch_size)
4. Validação formal de paridade numérica (torch.allclose e np.testing.assert_allclose)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def obter_modelo_exemplo() -> nn.Module:
    """
    Retorna uma ResNet-18 configurada para avaliação.
    Utiliza pesos pré-treinados se disponíveis ou arquitetura padrão.
    """
    try:
        pesos = models.ResNet18_Weights.DEFAULT
        modelo = models.resnet18(weights=pesos)
    except Exception:
        # Fallback offline para testes locais sem conexão
        modelo = models.resnet18(weights=None)
    modelo.eval()
    return modelo


def demonstrar_torchscript(modelo: nn.Module, diretorio_saida: Path) -> Path:
    """
    Executa o fluxo completo de compilação, persistência e inferência com TorchScript.
    """
    print("\n" + "=" * 60)
    print("🚀 1. COMPILAÇÃO PARA TORCHSCRIPT (C++ / LibTorch)")
    print("=" * 60)

    caminho_pt = diretorio_saida / "resnet18_traced.pt"

    # 1. Trava em modo eval()
    modelo.eval()

    # 2. Cria amostra dummy com formato [Batch, Canais, Altura, Largura]
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # 3. Rastreia o grafo computacional
    with torch.no_grad():
        modelo_traced = torch.jit.trace(modelo, dummy_input)

    # 4. Salva no disco
    modelo_traced.save(str(caminho_pt))
    print(f"📦 Modelo salvo em: {caminho_pt} ({caminho_pt.stat().st_size / (1024 * 1024):.2f} MB)")

    # 5. Carrega o modelo de volta (sem precisar do código da classe ResNet!)
    modelo_recarregado = torch.jit.load(str(caminho_pt))
    modelo_recarregado.eval()

    # 6. Validação de paridade numérica
    with torch.no_grad():
        saida_original = modelo(dummy_input)
        saida_traced = modelo_recarregado(dummy_input)

    diferenca = torch.max(torch.abs(saida_original - saida_traced)).item()
    print(f"📊 Diferença máxima absoluta: {diferenca:.8e}")
    assert torch.allclose(saida_original, saida_traced, atol=1e-5), "Falha na paridade do TorchScript!"
    print("✅ Sucesso: Grafo TorchScript idêntico ao modelo PyTorch original.")

    return caminho_pt


def demonstrar_onnx(modelo: nn.Module, diretorio_saida: Path) -> Path | None:
    """
    Executa a exportação para o padrão interoperável ONNX com suporte a batch dinâmico.
    """
    print("\n" + "=" * 60)
    print("🌐 2. EXPORTAÇÃO PARA ONNX (Open Neural Network Exchange)")
    print("=" * 60)

    caminho_onnx = diretorio_saida / "resnet18.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("⚠️ Pacote 'onnx' não encontrado no ambiente.")
        print("   Para exportar para ONNX, instale com: uv add onnx (ou pip install onnx)")
        return None

    modelo.eval()
    print("🔄 Exportando grafo ONNX com eixos dinâmicos...")
    torch.onnx.export(
        model=modelo,
        args=dummy_input,
        f=str(caminho_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"📦 Modelo salvo em: {caminho_onnx} ({caminho_onnx.stat().st_size / (1024 * 1024):.2f} MB)")

    # Teste de inferência com ONNX Runtime (se instalado)
    try:
        import onnxruntime as ort

        print("\n🔍 Testando inferência desvinculada via ONNX Runtime...")
        session = ort.InferenceSession(str(caminho_onnx), providers=["CPUExecutionProvider"])
        
        # Teste com lote dinâmico (Batch=4 em vez de 1!)
        lote_teste = torch.randn(4, 3, 224, 224, dtype=torch.float32)
        
        with torch.no_grad():
            ref_pytorch = modelo(lote_teste).numpy()

        ort_inputs = {session.get_inputs()[0].name: lote_teste.numpy()}
        ort_saida = session.run([session.get_outputs()[0].name], ort_inputs)[0]

        np.testing.assert_allclose(ref_pytorch, ort_saida, rtol=1e-3, atol=1e-5)
        print(f"✅ Sucesso: ONNX Runtime executou batch de {lote_teste.shape[0]} amostras com 100% de paridade!")

    except ImportError:
        print("ℹ️ Para testar a inferência sem PyTorch em produção: pip install onnxruntime")

    return caminho_onnx


def main() -> None:
    diretorio_saida = Path(__file__).resolve().parent / "artefatos_deploy"
    diretorio_saida.mkdir(exist_ok=True)

    print("Iniciando pipeline de exportação para produção...")
    modelo = obter_modelo_exemplo()

    # 1. TorchScript
    demonstrar_torchscript(modelo, diretorio_saida)

    # 2. ONNX
    demonstrar_onnx(modelo, diretorio_saida)

    print("\n🏁 Demonstração de exportação concluída com sucesso!")


if __name__ == "__main__":
    main()
