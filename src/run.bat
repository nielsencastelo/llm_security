@echo off
echo Executando LLM.
E:
cd "E:\Projetos\llm_security\src"
call conda activate GPU_inatel
streamlit run llm_security.py