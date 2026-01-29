# Import the PDF Loader instead of TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the PDF
print("Loading PDF...")
# We use PyPDFLoader to read 'document.pdf'
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Chunking (Splitting the text)
print(f"Splitting {len(documents)} pages of text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Embedding & Storing
print("Creating Vector Store...")
# We use the 'local_files_only' trick again to avoid network errors
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'local_files_only': True} 
)

# Save to ChromaDB
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model, 
    persist_directory="chroma_db"
)

print(f"Success! Saved {len(chunks)} chunks of knowledge to the database.")