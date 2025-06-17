# Smart Recruiters RAG System

## 📋 Overview

Smart Recruiter is an intelligent system that parses raw PDF CVs, transforms them into structured Markdowns and JSONs, embeds their content using LLMs, and stores them in a vector database (ChromaDB). It supports semantic search and retrieval to assist in recruitment workflows.

## ✨ Features

- **CV Parsing**: Extracts and structures information from unstructured PDF resumes
- **Embedding**: Uses `nomic-embed-text` via Ollama to embed the extracted data
- **Semantic Search**: Stores embeddings in ChromaDB for fast and relevant retrieval
- **LLM Integration**: Leverages `llama3:instruct` for reasoning and query generation
- **Modular**: Designed for easy extensibility and integration
- **Smart Matching**: Advanced candidate-job matching using RAG techniques
- **Metadata Extraction**: Intelligent extraction of skills, experience, and qualifications


## 💻 Usage

### Basic Usage

1. **Upload CVs**: Using Gradio UI or Place PDF resumes in the `data/pdf/` directory
2. **Search**: Use semantic search to find relevant candidates
3. **Match**: Get intelligent candidate-job matching results

## 🏗️ Architecture

The system follows a modular architecture with the following components:

### Core Components

- **Agents**: Specialized agents for different processing tasks
  - `markdown_agent.py`: Converts PDFs to structured markdown
  - `metadata_agent.py`: Extracts structured metadata from CVs

- **Model**: LLM integration and configuration
  - `model.py`: Main model interface
  - `model_config.py`: Model configuration and parameters

- **RAG**: Retrieval-Augmented Generation system
  - `rag.py`: Core RAG implementation for candidate matching

- **Utils**: Utility functions and database management
  - `prepare_data.py`: Data preprocessing utilities
  - `vectordb.py`: Vector database operations

## 📁 Project Structure

```
Smart-Recruiters-RAG-system/
│
├── app.py                     # UI entry point
├── main.py                    # Main application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── agents/                    # Processing agents
│   ├── markdown_agent.py      # PDF to markdown conversion
│   └── metadata_agent.py      # Metadata extraction
│
├── data/                      # Data storage
│   ├── markdowns/             # Generated CV markdown files
│   ├── metadata/              # JSON metadata for each CV
│   └── pdf/                   # Original PDF files
│
├── model/                     # LLM integration
│   ├── model.py               # Model interface
│   └── model_config.py        # Configuration settings
│
├── rag/                       # RAG system
│   └── rag.py                 # Core RAG implementation
│
└── utils/                     # Utilities
    ├── prepare_data.py        # Data preprocessing
    └── vectordb.py            # Vector database operations
```

## 🔧 Technologies Used

- **Python**: Core programming language
- **Ollama**: Local LLM hosting with `nomic-embed-text` and `llama3.2`
- **ChromaDB**: Vector database for embeddings storage
- **PyMuPDF / pdfminer / pdfplumber**: PDF parsing and text extraction
- **Hugging Face Transformers**: Additional ML model support (optional)
- **JSON**: Structured data storage format

## 👥 Authors

This project was developed by:
* [Amro Alaa kahla](https://github.com/Amrokahla)
* [Sarah Nabil Kamel](https://github.com/SarahNabilKamel)
* [Omar Ayman Elgemaey](https://github.com/OmarrAymann)
* [Alaa Khaled Seif](https://github.com/Aalaa25)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🛠️ Prerequisites

Make sure Ollama is installed and running on your system.

### Install Required Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amrokahla/Smart-Recruiters-RAG-system.git
   cd Smart-Recruiters-RAG-system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```


## 📝 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:
- Create an issue on [GitHub Issues](https://github.com/Amrokahla/Smart-Recruiters-RAG-system/issues)
- Check the documentation for common solutions
- Review existing issues for similar problems

**Happy Recruiting! 🎯**
