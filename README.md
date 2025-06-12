# 🧠 Smart Recruiter

**Smart Recruiter** is an intelligent system that parses raw PDF CVs, transforms them into structured JSON, embeds their content using LLMs, and stores them in a vector database (ChromaDB). It supports semantic search and retrieval to assist in recruitment workflows.

## 🚀 Features

* 📄 **CV Parsing**: Extracts and structures information from unstructured PDF resumes.
* 🧠 **Embedding**: Uses `nomic-embed-text` via Ollama to embed the extracted data.
* 🔍 **Semantic Search**: Stores embeddings in ChromaDB for fast and relevant retrieval.
* 🤖 **LLM Integration**: Leverages `llama3:instruct` for reasoning and query generation.
* 📦 **Modular**: Designed for easy extensibility and integration.

## 📦 Installation & Usage

### 1. Install Ollama and Pull Models

Make sure Ollama is installed and running.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-recruiter.git
cd smart-recruiter
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python main.py
```

## 🛠️ Tech Stack

* Python
* Ollama (with `nomic-embed-text` and `llama3:instruct`)
* ChromaDB
* PyMuPDF / pdfminer / pdfplumber (for PDF parsing)
* Hugging Face Transformers (optional)

## 🗂️ Project Structure

```
smart-recruiter/
│   main.py
│   README.md
│   requirements.txt
│
├───agents
│       markdown_agent.py
│       metadata_agent.py
│
├───data
│   ├───markdowns   # generated CV markdown
│   ├───metadata    # json metadata for each cv
│   └───pdf         # original file
│
├───model
│   │   model.py
│   │   model_config.py
│
├───rag
│       rag.py
│
└───utils
        prepare_data.py
        vectordb.py
```

## 👥 Contributors

This project was developed by:

* [Amro kahla](https://github.com/Amrokahla)
* [Sarah Nabil](https://github.com/SarahNabilKamel)
* [Omar Ayman](https://github.com/OmarrAymann)
* [Alaa Khaled](https://github.com/Aalaa25)