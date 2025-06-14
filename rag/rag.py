from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from collections import defaultdict


# === Load or Create Vector Store ===
def load_vectorstore(chroma_path, client, embedding_model="nomic-embed-text"):
    embedding = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding,
        collection_name="applicants",
        client=client
    )
    print(f"✅ Loaded collection with {vectorstore._collection.count()} chunks")
    return vectorstore


# === Custom Prompt Template for QA ===
def get_prompt_template():
    template = """
You are a recruitment assistant with access to candidate resumes.

Use the following candidate data to answer the question.
If the answer isn't available, respond with "I don't know".

Each document may include:
- Candidate name and email
- Skills
- Projects
- Education
- Work history

Context:
{context_with_metadata}

Question:
{question}
"""
    return PromptTemplate(
        input_variables=["context_with_metadata", "question"],
        template=template
    )


# === Build RAG Chain ===
def build_rag_chain(vectorstore, llm_model="llama3.2"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOllama(model=llm_model)
    prompt = get_prompt_template()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context_with_metadata"
        },
        return_source_documents=True
    )
    return chain


# === Interactive Q&A Terminal Loop ===
def run_qa_loop(qa_chain):
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa_chain.invoke(query)
        answer = result["result"]
        source_documents = result["source_documents"]

        context_with_metadata = "\n\n".join(
            f"Content: {doc.page_content}\nMetadata: {doc.metadata}"
            for doc in source_documents
        )

        print("\nAnswer:", answer)


# === Find Top Candidates Based on JD ===
def find_top_candidates(job_description, vectorstore, top_k=3):
    results = vectorstore.similarity_search(job_description, k=10)
    candidate_scores = defaultdict(list)
    candidate_chunks = defaultdict(list)

    for doc in results:
        meta = doc.metadata
        key = (meta.get("candidate_name"), meta.get("candidate_email"))
        candidate_scores[key].append(1)
        candidate_chunks[key].append(doc.page_content)

    ranked = sorted(candidate_scores.items(), key=lambda x: len(x[1]), reverse=True)

    output = []
    for i, ((name, email), _) in enumerate(ranked[:top_k], 1):
        info = candidate_chunks[(name, email)][:2]  # return top 2 relevant chunks
        output.append(((name, email), info))

    return output


# === Summarize Candidate CV by Name ===
def summarize_candidate_cv(candidate_name: str, vectorstore, llm_model="llama3.2"):
    llm = ChatOllama(model=llm_model)
    all_docs = vectorstore.get()
    
    # Normalize name
    candidate_name = candidate_name.replace("_", " ").replace(".md", "").strip().lower()
    if not candidate_name:
        print("❌ Candidate name cannot be empty.")
        return None
    if len(candidate_name) < 3:
        print("❌ Candidate name must be at least 3 characters long.")
        return None
    if not all_docs or "documents" not in all_docs or not all_docs["documents"]:
        print("❌ No documents found in the vector store.")
        return None
    
    # Match documents by content or metadata
    matched_docs = [
        doc for doc in all_docs["documents"]
        if doc and candidate_name.lower() in doc.lower()
    ]

    matched_metadata = []
    if not matched_docs:
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            if meta.get("candidate_name", "").lower() == candidate_name.lower():
                matched_docs.append(doc)
                matched_metadata.append(meta)

    if not matched_docs:
        print(f"❌ No documents found for candidate: {candidate_name}")
        return None

    documents = [Document(page_content=doc) for doc in matched_docs]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(documents)

    email = "unknown@example.com"
    for meta in all_docs["metadatas"]:
        if meta.get("candidate_name", "").lower() == candidate_name.lower():
            email = meta.get("candidate_email", email)
            break

    return {
        "summary": summary,
        "email": email
    }
