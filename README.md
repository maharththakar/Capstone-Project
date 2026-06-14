# Legal AI Navigator

Automated pipeline for classifying legal PDF communications using LangGraph and Llama-3.1-8B.

## Setup

### System Dependencies

#### macOS

```bash
brew install tesseract poppler
```

#### Linux

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### Python Environment

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY="your_groq_api_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="Capstone_Classification_Pipeline"
```

## Execution

Launch the Streamlit application:

```bash
streamlit run app.py
```
