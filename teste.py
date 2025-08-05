import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar dados brutos
df = pd.read_parquet('test_net.parquet')
features_to_use = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']

preprocessor = joblib.load('preprocessor_model.joblib')
modelo = joblib.load('gradient_boosting_model.joblib')

# Função para renderizar "badge" colorido
def render_badge(label, mapping={0: 'Normal', 1: 'Ataque'}):
    color = '#13c26b' if label == 0 else '#ff2c2c'
    texto = mapping[label]
    st.markdown(
        f'<div style="display:inline-block;'
        f'background-color:{color};color:white;'
        f'border-radius:20px;padding:4px 18px;'
        f'font-weight:bold;font-size:20px;margin:3px;">{texto}</div>',
        unsafe_allow_html=True
    )

st.set_page_config(layout="wide", page_title="Monitoramento de Ataques de Rede")

st.markdown(
    "<h1 style='text-align: center; color: #003366;'>Dashboard de Monitoramento de Pacotes - Machine Learning</h1>",
    unsafe_allow_html=True
)

st.write("---")

if 'i' not in st.session_state:
    st.session_state.i = 0

col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    st.markdown("### Controles")
    if st.button('Próximo Pacote'):
        st.session_state.i += 1
    if st.button('Pacote Anterior') and st.session_state.i > 0:
        st.session_state.i -= 1

with col2:
    idx = st.session_state.i
    if idx < len(df):
        row = df.iloc[idx]
        features = preprocessor.transform(row[features_to_use].to_frame().T)
        pred = modelo.predict(features)[0]
        real = int(row['binary_label']) if 'binary_label' in row else None

        st.markdown(f"#### <center>Pacote Atual: <b>{idx}</b></center>", unsafe_allow_html=True)
        st.progress((idx + 1) / len(df))

        st.write("**Características do pacote:**")
        st.dataframe(row[features_to_use].to_frame().T, use_container_width=True)

        st.write("**Resultado da Classificação:**")
        colreal, colpred = st.columns(2)
        with colreal:
            st.markdown("#### Real")
            render_badge(real)
        with colpred:
            st.markdown("#### Predição")
            render_badge(pred)
    else:
        st.warning("Fim do dataset!")

with col3:
    st.markdown("### Estatísticas Gerais")
    total = idx + 1
    n_normal = int(sum(df.iloc[:total]['binary_label'] == 0))
    n_ataque = int(sum(df.iloc[:total]['binary_label'] == 1))
    perc_normal = 100 * n_normal / total if total > 0 else 0
    perc_ataque = 100 * n_ataque / total if total > 0 else 0

    st.metric(label="Total exibidos", value=f"{total}")
    st.metric(label="Normais", value=f"{n_normal}", delta=f"{perc_normal:.1f} %")
    st.metric(label="Ataques", value=f"{n_ataque}", delta=f"{perc_ataque:.1f} %")

    st.markdown(
        f"<span style='color:#13c26b;font-weight:bold;'>Normais:</span> {perc_normal:.1f}% &nbsp;&nbsp;&nbsp; "
        f"<span style='color:#ff2c2c;font-weight:bold;'>Ataques:</span> {perc_ataque:.1f}%",
        unsafe_allow_html=True
    )

    st.write("---")
    st.markdown(
        "<p style='font-size:12px;color:gray;text-align:right;'>Powered by Streamlit + Scikit-learn</p>",
        unsafe_allow_html=True
    )

