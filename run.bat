@echo off
echo Executando LLM.
E:
cd "E:\Inatel\Projetos\oai-anomaly-detection\notebooks"
call conda activate GPU_inatel
streamlit run llm_security.py