import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ---- CONFIGURATION ----
DOCS_FOLDER = "./my_documents"       # Put your files here
DB_FOLDER = "./vector_db"            # Where the searchable index is saved
MODEL_NAME = "llama3.2"              # The Ollama model you downloaded

os.makedirs(DOCS_FOLDER, exist_ok=True)

# Load and process documents
def load_documents():
    docs = []
    for file in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def build_index():
    print("📚 Reading your documents...")
    docs = load_documents()
    
    # Split into chunks (easier to search small pieces than whole docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} searchable chunks")

    if not chunks:
        print("⚠️  No documents found! Please add .pdf or .txt files to the './my_documents' folder.")
        return None

    # Convert text chunks to numbers (embeddings) for semantic search
    print("🔢 Creating embeddings (this may take a few minutes)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Store in local ChromaDB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)
    print("✅ Index built and saved!")
    return db

def get_qa_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Grab top 3 relevant chunks
    llm = OllamaLLM(model=MODEL_NAME)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Build index on startup
build_index()
qa_chain = get_qa_chain()

# Chat function
def chat(question, history):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = set(doc.metadata.get("source", "unknown") for doc in result["source_documents"])
    answer += f"\n\n📎 *Sources: {', '.join(sources)}*"
    return answer

# Launch UI
gr.ChatInterface(
    fn=chat,
    title="📄 Local RAG Agent",
    description="Ask questions about your documents. Everything stays on your computer!",
).launch()