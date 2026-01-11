# 📄 Resume Analyzer Chatbot

A **smart resume analysis application** that allows you to upload your resume in **PDF format** and ask questions about its content using **AI-powered Retrieval-Augmented Generation (RAG)**.

This tool helps you quickly understand, analyze, and query resumes (or any PDF document) with accurate, context-aware answers.

---

## ✨ Features

- 📄 **PDF Upload & Processing**  
  Upload any PDF with automatic text extraction and intelligent chunking.

- 🔍 **Intelligent Search**  
  Uses vector embeddings to retrieve the most relevant content from your document.

- 🤖 **AI-Powered Answers**  
  Get precise, context-based answers using **Groq LLM**.

- 💬 **Chat History**  
  Maintains conversation history within a session.

- ⚙️ **Customizable Settings**  
  Adjust temperature and retrieval parameters for better control.

- 🎨 **Clean UI**  
  Simple and intuitive **Streamlit** interface.

- 🔄 **Multi-Document Support**  
  Upload and analyze different PDFs without data mixing.

---

## 🛠️ Tech Stack

| Component | Technology |
|---------|-----------|
| Frontend | Streamlit |
| LLM Provider | Groq (`openai/gpt-oss-120b`) |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| PDF Processing | LangChain Community (`PyPDFLoader`) |
| Framework | LangChain (LCEL) |
| Language | Python 3.8+ |

---

## 📋 Prerequisites

- Python **3.8 or higher**
- Groq API Key (free tier available)
- Basic understanding of Python and command line

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-qa-assistant.git
cd pdf-qa-assistant
```

### 2️⃣ Create a Virtual Environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.0
langchain-groq>=0.0.1
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
python-dotenv>=1.0.0
chromadb>=0.4.18
pypdf>=3.17.0
sentence-transformers>=2.2.0
```

---

## 🎯 Usage

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

---

## 📁 Project Structure

```
pdf-qa-assistant/
│
├── streamlit_app.py
├── .env
├── requirements.txt
├── README.md
│
├── uploads/
├── chroma_db_*/
└── .gitignore
```

---
