import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

# --- 1. CONFIGURATION ---
# Load key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)




# --- 2. SETUP MEMORY (Local Files) ---
print("Loading memory...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'local_files_only': True} 
)
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# --- 3. THE QUESTION ---
query = "how many fixed holidays are there ?"
print(f"\nUser Question: {query}")

# --- 4. RETRIEVAL ---
# Get top 3 chunks
results = db.similarity_search(query, k=3)
context_text = "\n\n".join([doc.page_content for doc in results])

print(f"\n--- 🔍 RETRIEVED CONTEXT ---")
print(f"{context_text[:500]}...") 
print("----------------------------\n")

# --- 5. GENERATION (Direct Google Call) ---
prompt = f"""
Answer the question based ONLY on the following context.
Count the items carefully.

Context:
{context_text}

Question:
{query}

Answer:
"""


print("Asking Gemini (Directly)...")

# We use a verified model from the available list
model = genai.GenerativeModel('models/gemini-flash-latest')
response = model.generate_content(prompt)
print("\n=== 💎 GEMINI ANSWER ===")
print(response.text)
print("========================")