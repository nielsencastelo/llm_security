import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

PROMPT_FILE = r"E:\Projetos\llm_security\src\Prompt - secur.txt"
path_dataset = r'E:\Projetos\llm_security\dataset'
path_models = r'E:\Projetos\llm_security\models'

MODEL_NAME = "phi4"

# Carregar prompt do sistema (especialista em segurança)
with open(PROMPT_FILE, encoding='utf-8') as f:
    system_prompt = f.read()

def load_llm():
    if 'llm_instance' not in st.session_state:
        llm = ChatOllama(model=MODEL_NAME.lower().replace(" ", ""), temperature=0.1)
        st.session_state.llm_instance = llm

def model_response(user_query, chat_history=None):
    try:
        llm = st.session_state.llm_instance
        # Começa com o prompt do sistema
        messages = [("system", system_prompt)]
        # Adiciona todo o histórico, se existir
        if chat_history:
            messages += chat_history
        # Adiciona a nova pergunta do usuário
        messages.append(("user", user_query))
        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke({"input": user_query})
    except Exception as e:
        print("Erro:", e)
        return "❌ Ocorreu um erro ao processar sua solicitação."


# Carregar LLM ao abrir o app
load_llm()


df = pd.read_parquet(f'{path_dataset}/test_net.parquet')
features_to_use = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
preprocessor = joblib.load(f'{path_models}/preprocessor_model.joblib')
modelo = joblib.load(f'{path_models}/gradient_boosting_model.joblib')

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
        # Ao mudar o pacote, resetar histórico do LLM para novo pacote
        st.session_state.pop("llm_chat_history", None)
        st.session_state.pop("llm_chat_idx", None)
        st.session_state.pop("llm_ja_avaliado", None)
    if st.button('Pacote Anterior') and st.session_state.i > 0:
        st.session_state.i -= 1
        st.session_state.pop("llm_chat_history", None)
        st.session_state.pop("llm_chat_idx", None)
        st.session_state.pop("llm_ja_avaliado", None)

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

#### ==== SIDEBAR COM BOTÃO E CHAT LLM SOBRE O ATAQUE ==== ####
if idx < len(df):
    row = df.iloc[idx]
    pred = modelo.predict(preprocessor.transform(row[features_to_use].to_frame().T))[0]
    if pred == 1:  # Se classificado como ataque
        st.sidebar.title("🛡️ Análise com Especialista LLM")
        st.sidebar.write("**Pacote Selecionado:**")
        st.sidebar.dataframe(row[features_to_use].to_frame().T)
        st.sidebar.write(f"**Classificação:** {row['label'] if 'label' in row else 'Ataque'}")

        pacote_str = ", ".join(str(x) for x in row.values)

        # Botão para acionar análise LLM do pacote
        if "llm_ja_avaliado" not in st.session_state or st.session_state.llm_chat_idx != idx:
            if st.sidebar.button("Avaliar com especialista LLM"):
                exp_llm = model_response(pacote_str)
                st.session_state.llm_chat_history = [("system", exp_llm)]
                st.session_state.llm_chat_idx = idx
                st.session_state.llm_ja_avaliado = True
        else:
            # Já foi avaliado, mostra chat
            for who, msg in st.session_state.llm_chat_history:
                st.sidebar.markdown(f"**{'LLM' if who == 'system' else 'Você'}:**")
                st.sidebar.markdown(msg)

            # user_ask = st.sidebar.text_input("Pergunte ao especialista LLM sobre este ataque:", key=f"llm_ask_{idx}")
            # if st.sidebar.button("Enviar pergunta", key=f"send_{idx}") and user_ask:
            #     user_input = f"{user_ask}\n\nContexto do pacote: {pacote_str}"
            #     resp = model_response(user_input)
            #     st.session_state.llm_chat_history.append(("user", user_ask))
            #     st.session_state.llm_chat_history.append(("system", resp))
            #     # st.experimental_rerun()
            #     st.rerun()
            user_ask = st.sidebar.text_input("Pergunte ao especialista LLM sobre este ataque:", key=f"llm_ask_{idx}")
            if st.sidebar.button("Enviar pergunta", key=f"send_{idx}") and user_ask:
                user_input = f"{user_ask}\n\nContexto do pacote: {pacote_str}"
                # Monta o histórico no formato [(role, msg), ...] sem repetir o system
                chat_hist = [x for x in st.session_state.llm_chat_history if x[0] != "system"]
                resp = model_response(user_input, chat_history=chat_hist)
                st.session_state.llm_chat_history.append(("user", user_ask))
                st.session_state.llm_chat_history.append(("system", resp))
                st.rerun()


    else:
        st.sidebar.info("Selecione um pacote de ataque para abrir o chat com o especialista LLM.")
