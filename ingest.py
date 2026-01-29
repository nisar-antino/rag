import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from glob import glob

DATA_PATH = "data"
DB_PATH = "chroma_db"

def ingest_docs():
    documents = []
    
    # 1. Load PDFs
    print("Scanning for PDFs...")
    pdf_files = glob(f"{DATA_PATH}/*.pdf")
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    # 2. Load TXT files
    print("Scanning for Text files...")
    txt_files = glob(f"{DATA_PATH}/*.txt")
    for txt_file in txt_files:
        print(f"Loading {txt_file}...")
        loader = TextLoader(txt_file)
        documents.extend(loader.load())

    if not documents:
        print("No documents found in data/ folder!")
        return

    print(f"Loaded {len(documents)} document pages/files.")

    # 3. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 4. Save to Chroma
    print("Saving to Vector Database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'local_files_only': True}
    )
    
    # Add to existing DB or create new
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=DB_PATH
    )
    
    print("✅ Success! Database updated.")

if __name__ == "__main__":
    ingest_docs()
