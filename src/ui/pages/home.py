import streamlit as st

st.title("AgroVision — Classificação Foliar")
st.markdown(
    "Sistema de Visão Computacional para **identificação precoce de patologias em folhas** "
    "utilizando Redes Neurais Convolucionais (CNNs) implementadas em PyTorch."
)

st.divider()

# Apresentação das Patologias
st.subheader("Patologias Diagnosticadas")

col_h, col_p, col_r = st.columns(3)

with col_h:
    with st.container(border=True):
        st.markdown("#### Saudável (Healthy)")
        st.markdown(
            """
            - **Aspecto**: Coloração verde uniforme, sem lesões, clorose ou pústulas.
            - **Impacto**: Fotossíntese e desenvolvimento vegetal plenos.
            - **Manejo**: Manutenção do programa preventivo de nutrição e irrigação.
            """
        )

with col_p:
    with st.container(border=True):
        st.markdown("#### Oídio (Powdery Mildew)")
        st.markdown(
            """
            - **Aspecto**: Camada pulverulenta esbranquiçada ou acinzentada sobre a lâmina foliar.
            - **Impacto**: Redução da taxa fotossintética e enfraquecimento do tecido.
            - **Manejo**: Controle de umidade, aeração e aplicação de fungicidas recomendados.
            """
        )

with col_r:
    with st.container(border=True):
        st.markdown("#### Ferrugem (Rust)")
        st.markdown(
            """
            - **Aspecto**: Pústulas salientes de cor amarelo-alaranjada a castanho-ferrugem.
            - **Impacto**: Ruptura da epiderme foliar, perda de água e desfolha precoce.
            - **Manejo**: Intervenção rápida com fungicidas sistêmicos e isolamento de focos.
            """
        )

st.divider()

# Informações Essenciais da Base e Arquitetura
col_data, col_arch = st.columns(2)

with col_data:
    st.subheader("Base de Dados")
    st.markdown(
        """
        Conjunto balanceado contendo **1.532 imagens** distribuídas em:
        - **Treino**: 1.322 imagens
        - **Validação**: 60 imagens
        - **Teste**: 150 imagens
        """
    )

with col_arch:
    st.subheader("Arquiteturas Disponíveis")
    st.markdown(
        """
        Modelos com **Transfer Learning & Fine-Tuning em 2 Fases**:
        - **ResNet-50**: Backbone residual profundo com 50 camadas (~98% acurácia no teste).
        - **MobileNetV3-Small**: Otimizado para alta performance e baixa latência em borda.
        - **Pipeline Orientado a Objetos**: Modelos padronizados com contrato `BaseModel`.
        """
    )

st.divider()

st.info("Para testar uma imagem, acesse a aba **Diagnóstico de Folha** no menu lateral.")
