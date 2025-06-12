import warnings
warnings.filterwarnings("ignore")

import os
from utils.prepare_data import generate_markdowns, generate_metadata
from chromadb import Client
from chromadb.config import Settings
from utils.vectordb import generate_data_store
from rag.rag import load_vectorstore, build_rag_chain, find_top_candidates, summarize_candidate_cv
from langchain_ollama import OllamaEmbeddings

client_settings = Settings(persist_directory="chroma_store")
shared_client = Client(client_settings)

def prepare_data():
    parent_path = "data"
    if not os.path.exists(parent_path):
        print(f"Error: The directory '{parent_path}' does not exist.")
        return
    generate_markdowns(parent_path)
    generate_metadata(parent_path)

    print("Markdown/metadata files generated successfully.")

def intialize_chromadb():
    data_path = "data/markdowns"
    metadata_path = "data/metadata"
    db_path = os.path.join(os.getcwd(), "chroma_store")
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    generate_data_store(data_path, metadata_path ,"chroma_store", embedding_function, shared_client)
    

def run_rag(vectorstore):
    qa_chain = build_rag_chain(vectorstore)

    while True:
        mode = input("\nChoose mode: [q] Q&A | [j] Job Match | [exit]: ").lower()

        if mode == "exit":
            break
        elif mode == "q":
            query = input("Enter your question: ")
            result = qa_chain.invoke(query)
            print("\nAnswer:", result["result"])
        elif mode == "j":
            job_description = input("Enter job description: ")
            find_top_candidates(job_description, vectorstore)
        else:
            print("Invalid option.")

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
            find_top_candidates(job_desc, vectorstore)

        elif choice == "2":
            query = input("\nAsk your question: ")
            result = qa_chain.invoke(query)
            print("\nAnswer:", result["result"])

        elif choice == "3":
            name = input("Enter candidate name: ")
            summarize_candidate_cv(name, vectorstore)

        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")



if __name__ == "__main__":
    parent = 'data'
    prepare_data()
    intialize_chromadb()
    vectorstore = load_vectorstore("chroma_store", shared_client)
    qa_chain = build_rag_chain(vectorstore)
    run_interactive_menu(vectorstore, qa_chain)


