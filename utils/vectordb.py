from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os, json
from pathlib import Path

def load_documents(md_path, json_path):
    documents = []

    for md_file in os.listdir(md_path):
        if md_file.endswith('.md'):
            base_name = Path(md_file).stem
            json_file = os.path.join(json_path, f"{base_name}.json")
            md_file_path = os.path.join(md_path, md_file)

            with open(md_file_path, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = {"source": md_file_path}
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as json_f:
                    metadata.update(json.load(json_f))

            documents.append(Document(page_content=text, metadata=metadata))

    return documents

def split_text(documents):
    return documents

def save_to_chroma(chunks, chroma_path, embedding_function, client):
    print("🚀 Saving to ChromaDB...")
    try:
        db = Chroma.from_documents(
            chunks,
            embedding_function,
            persist_directory=chroma_path,
            client=client,
            collection_name="applicants"
        )
        print("✅ Finished saving to Chroma.")
    except Exception as e:
        print("❌ Error saving to Chroma:", e)


def generate_data_store(md_path, json_path, chroma_path, embedding_function, client):
    documents = load_documents(md_path, json_path)
    chunks = split_text(documents)
    if not chunks:
        print("❌ No chunks to save — exiting.")
        return
    save_to_chroma(chunks, chroma_path, embedding_function, client)
    print(f"✅ Data store saved to {chroma_path}")