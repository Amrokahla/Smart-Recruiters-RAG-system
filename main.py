import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import tempfile
from chromadb import Client
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from utils.prepare_data import generate_markdowns, generate_metadata
from utils.vectordb import generate_data_store
from rag.rag import (
    load_vectorstore,
    build_rag_chain,
    find_top_candidates,
    summarize_candidate_cv
)

# === Global ChromaDB client ===
client_settings = Settings(persist_directory="chroma_store")
shared_client = Client(client_settings)

# === Step 1: Prepare Markdown and Metadata Files ===
def prepare_data():
    print("✅ Skipped saving markdown/metadata to disk. Data will be processed in-memory when CVs are uploaded.")

# === Step 2: Initialize Chroma Vector Store ===
def intialize_chromadb():
    print("✅ ChromaDB will be populated dynamically after CV upload.")

# === Step 3: CLI Interface ===
def run_interactive_menu(vectorstore, qa_chain):
    while True:
        print("\nChoose an action:")
        print("1. Find top candidates for a job description")
        print("2. Ask a question about candidates")
        print("3. Summarize a candidate's CV")
        print("4. Exit")
        choice = input("Enter 1, 2, 3, or 4: ").strip()

        if choice == "1":
            job_desc = input("\nEnter job description: ")
            results = find_top_candidates(job_desc, vectorstore)
            for i, ((name, email), chunks) in enumerate(results, 1):
                print(f"\n{i}. {name} ({email})")
                for chunk in chunks:
                    print(f"- {chunk[:200]}...")

        elif choice == "2":
            query = input("\nAsk your question: ")
            result = qa_chain.invoke(query)
            print("\nAnswer:", result["result"])

        elif choice == "3":
            name = input("Enter candidate name: ")
            summary = summarize_candidate_cv(name, vectorstore)
            if summary:
                print(f"\nCandidate Email: {summary['email']}")
                print(f"Summary:\n{summary['summary']}")

        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

# === Upload & Ingest New CV ===
def process_uploaded_cv(file_path, vectorstore, auto_delete_pdf=True):
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.basename(file_path)
        target_path = os.path.join(temp_dir, filename)
        shutil.copyfile(file_path, target_path)

        # Generate markdown and metadata into memory using temp_dir
        generate_markdowns(temp_dir)
        generate_metadata(temp_dir)

        # Embed the new CV into vectorstore
        generate_data_store(
            os.path.join(temp_dir, "markdowns"),
            os.path.join(temp_dir, "metadata"),
            "chroma_store",
            OllamaEmbeddings(model="nomic-embed-text"),
            vectorstore._client
        )

    # Use filename (minus extension) as candidate name
    candidate_name = filename.rsplit('.', 1)[0]
    summary_data = summarize_candidate_cv(candidate_name, vectorstore)
    if summary_data is None:
        return candidate_name, "unknown@example.com", "No summary available."
    return candidate_name, summary_data["email"], summary_data["summary"]

# === Main Execution ===
if __name__ == "__main__":
    prepare_data()
    intialize_chromadb()
    vectorstore = load_vectorstore("chroma_store", shared_client)
    qa_chain = build_rag_chain(vectorstore)
    run_interactive_menu(vectorstore, qa_chain)
