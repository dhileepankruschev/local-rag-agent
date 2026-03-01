# Local RAG Agent 📄

A purely local Retrieval-Augmented Generation (RAG) agent that lets you "chat" with your documents (`.pdf` or `.txt`) keeping all your data strictly on your own machine. It uses [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Ollama](https://ollama.com/), and [Gradio](https://gradio.app/) for a sleek web interface.

## Prerequisites

Before running the agent, make sure you have the following installed on your system:

1. **Python 3.10+**: Make sure Python is installed and accessible from your terminal.
2. **Ollama**: Download and install [Ollama](https://ollama.com/download) to run local language models.

### Pull Required Local Models

Our script is configured to use two specific models running locally on Ollama. Open a new terminal and pull them:

```bash
# Language model used for generating answers
ollama pull llama3.2

# Embedding model used to convert text into searchable numbers
ollama pull nomic-embed-text
```

### Python Environment Setup

It is highly recommended to run this inside a virtual environment.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (Windows)
.\venv\Scripts\activate
# (Mac/Linux: source venv/bin/activate)

# 3. Install required packages
pip install gradio langchain-community langchain-text-splitters langchain-chroma langchain-ollama langchain-classic pypdf chromadb
```

## How to Use & Test

1. **Add Documents**: Create a folder named `my_documents` in the root directory. Drop the `.pdf` or `.txt` files you want the AI to read into this folder.
   
2. **Run the Agent**: Start the agent by running the python script:
   ```bash
   python rag_agent.py
   ```

3. **Chat!**: The script will read your documents, build a searchable semantic index locally, and then launch a web server. Open your browser and navigate to:
   **`http://127.0.0.1:7860`**
   
4. **Ask Questions**: Ask questions based on the documents you provided. The response will include citations of the source chunks it used to formulate the answer.
