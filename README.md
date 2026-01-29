# Professional RAG Pipeline 🤖📚

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you chat with your own documents (PDFs, Text files) using Google's Gemini AI.

## 🌟 Features
- **Multi-Document Support**: Ingest multiple PDFs and TXT files at once.
- **Smart Retrieval**: Uses `ChromaDB` and `HuggingFace` embeddings to find the exact context.
- **AI Powered**: Uses Google's `Gemini-1.5-Flash` (via `google-genai`) for accurate, natural language answers.
- **Secure**: API keys are managed safely via `.env`.

---

## 🚀 Setup Guide

### 1. Prerequisites
- Python 3.10+ installed.
- A Google API Key (from [Google AI Studio](https://aistudio.google.com/)).

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/nisar-antino/rag.git
cd rag
python -m venv venv
# Activate venv:
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
1. Create a `.env` file in the root folder.
2. Add your Google API Key:
   ```env
   GOOGLE_API_KEY=AIzaSy...YourKeyHere
   ```

---

## 📖 Usage

### Step 1: Add Your Data
Place your documents (PDFs or Text files) inside the **`data/`** folder.
> *Example: `data/holiday_calendar.pdf`, `data/notes.txt`*

### Step 2: Build the Database (Ingest)
Run the ingestion script to scan your `data/` folder and update the vector memory:

```bash
python ingest.py
```
*Output: `✅ Success! Database updated.`*

### Step 3: Ask Questions
Run the main script to query your documents:

```bash
python final_rag.py
```
*Note: You can modify the `query` variable in `final_rag.py` to change your question.*

---

## 📂 Project Structure
- **`final_rag.py`**: The main chatbot script (Retrieval + Generation).
- **`ingest.py`**: Handles loading documents and updating the database.
- **`data/`**: Folder for your source documents.
- **`chroma_db/`**: Local vector database (stores the document memory).
- **`.env`**: Stores secrets (not uploaded to GitHub).
