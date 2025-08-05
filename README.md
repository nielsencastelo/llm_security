
# LLM Security – Intelligent Network Packet Analysis with Security Expert LLM

## Project Objective

This project aims to combine classic Machine Learning techniques with state-of-the-art Large Language Models (LLM) for the detection, explanation, and interactive analysis of network attacks, all through an interactive Streamlit dashboard.

The system:
- Automatically classifies network packets as "normal" or "attack" using pre-trained supervised ML models.
- Allows the user to interact via chat with an LLM (phi4, served locally with Ollama) acting as a security expert, explaining detected attacks, recommending countermeasures, and answering user questions about each analyzed packet.

## Architecture and Technologies

- **Python 3.10+**
- **Streamlit**: Interactive web dashboard.
- **Scikit-learn** and **joblib**: Model training, serialization, and inference (ML and preprocessing pipelines).
- **LangChain**: Integration and orchestration of LLM chat.
- **Ollama**: Local backend for serving open-source LLMs (`phi4` model used here).
- **Parquet, pandas, numpy**: Data handling and transformation.

## How to Use

1. **Prerequisites**
   - Python 3.10+ installed.
   - [Ollama](https://ollama.com/) installed and running locally.
   - Download the phi4 model:
     ```
     ollama pull phi4
     ```

2. **Install dependencies**
   It is highly recommended to use a virtual environment.

   ```
   pip install -r requirements.txt
   ```
   Or manually:
   ```
   pip install streamlit pandas numpy joblib scikit-learn langchain langchain_community langchain-core
   ```

3. **Run the Streamlit dashboard**
   ```
   streamlit run llm_security.py
   ```
   The dashboard will open in your browser, showing packet navigation, classification, statistics, and the button to trigger the LLM expert analysis.

## Data Structure and Scripts

- **test_net.parquet**: Network packet data with labels ("normal" or "attack").
- **preprocessor_model.joblib / preprocessor.pkl**: Preprocessing pipeline (e.g., ColumnTransformer), trained and serialized.
- **gradient_boosting_model.joblib**: Trained classification model.
- **Prompt - secur.txt**: System prompt for the LLM to act as a security expert.
- **llm_security.py**: Main interactive Streamlit script.

## Dashboard Features

- **Packet navigation**: Buttons to browse to next/previous packet.
- **Automatic classification**: Displays both the true and predicted ML labels.
- **Real-time statistics**: Shows totals, percentages, and color-coded badges.
- **LLM Security Expert Chat**:
  - Available for packets classified as "attack".
  - "Evaluate with LLM Expert" button triggers LLM analysis.
  - Multi-turn chat: keeps a chat history while viewing the same packet.

## Tests Performed

- Manual review of labeled packets and comparison of predicted vs. true labels.
- Performance testing of the LLM chat, including context-aware answers and multi-turn dialogue.
- Full integration test: navigation + classification + LLM chat interaction.
- Environment testing on Windows and Conda.

## Main Scripts

- **llm_security.py**: Main Streamlit app, loads models and manages packet navigation/chat.
- **Auxiliary notebooks/scripts**: For data processing, ML model training, validation, and exporting `.joblib`/`.parquet` files.
- **Prompt - secur.txt**: Editable prompt file for the LLM system role.

## Usage Example

When clicking "Evaluate with LLM Expert" for an attack packet, the user can:
- Get a detailed explanation of the attack, risks, and countermeasures.
- Ask further questions via chat, such as “How to configure pfSense to mitigate this attack?” and receive context-aware, expert-level answers from phi4.

## Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

---

Project developed by Nielsen Castelo.
