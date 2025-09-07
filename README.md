# 📄 LightRag_for_pdfs

A Python-based pipeline to ingest multiple PDF documents and query them using a custom GPT-powered Retrieval-Augmented Generation (RAG) system built with [LightRAG](https://github.com/light-rag/light-rag).

---

## 🚀 Features

- Extracts text from multiple PDFs
- Embeds documents using OpenAI embeddings
- Queries documents using GPT-4o mini
- Supports multiple query modes: naive, local, global, hybrid
- Rotating log file support
- Modular and async-ready

---

## 📁 Project Structure
LightRag_for_pdfs/ 
├── main.py # Main pipeline script 
├── lightrag_demo.py # Additional demo or utility script 
├── requirements.txt # Python dependencies 
├── .env # API keys (not tracked in Git) 
├── .gitignore # Git ignore rules 
├── /dickens # Working directory for LightRAG 
├── /logs # Log files (auto-created) 
├── /pdfs # Place your PDF files here 
├── /venv # Virtual environment (optional)

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/LightRag_for_pdfs.git
cd LightRag_for_pdfs
```

### 2. Create and Activate Virtual Environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```
create a .env file
OPENAI_API_KEY=your-openai-api-key
LOG_DIR=./logs
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
VERBOSE_DEBUG=false
```

### 5. Add PDF Files
```
Place your PDF files inside the /pdfs folder.
```

### 6. Run your file
```
python main.py
```
