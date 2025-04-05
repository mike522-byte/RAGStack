# RAGStack
A Retrieval-Augmented Generation (RAG) system with local LLM support.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to Huggingface (if downloading models):
```bash
huggingface-cli login
```

3. Download a model:
```bash
# Set your Hugging Face token
export HUGGINGFACE_TOKEN="your_token_here"
python app/download_model.py "your_model_here"
```

## Running the System

### Option 1: Run everything with a single command

```bash
# Basic usage
python run.py

# With custom model and 4-bit quantization
python run.py --model-name mistral-7b-instruct-v0.2 --use-4bit
```

### Option 2: Run services separately

```bash
# Terminal 1 - Start the API
uvicorn app.api:app --reload --port 8000

# Terminal 2 - Start the Streamlit interface
streamlit run app/main.py
```

## Accessing the System

- Frontend: http://localhost:8501
- API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Features

- Multiple Documents upload and processing
- Semantic search and retrieval
- Local LLM inference
- 4-bit quantization support

## Configuration

You can configure the system using environment variables:

- `MODEL_NAME`: Name of the model to use (e.g., "mistral-7b-instruct-v0.2")
- `USE_4BIT`: Whether to use 4-bit quantization (true/false)

Or through the command line arguments when using `run.py`.
