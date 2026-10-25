from pathlib import Path
import streamlit as st

# Diretório base das páginas
PAGES_DIR = Path(__file__).resolve().parent / "pages"

# Configuração global da aplicação
st.set_page_config(
    page_title="AgroVision - Classificação Foliar",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definição das páginas
home_page = st.Page(
    PAGES_DIR / "home.py",
    title="Visão Geral",
    default=True,
)

predict_single_page = st.Page(
    PAGES_DIR / "predict_single.py",
    title="Diagnóstico de Folha",
)

# Navegação estruturada
pg = st.navigation(
    {
        "Navegação": [home_page, predict_single_page],
    }
)

# Barra lateral
with st.sidebar:
    st.markdown("### AgroVision")
    st.caption("Classificação de patologias em folhas")
    st.divider()

pg.run()