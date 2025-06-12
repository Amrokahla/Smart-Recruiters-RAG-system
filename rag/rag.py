from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from collections import defaultdict
from langchain.prompts import PromptTemplate


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
    return PromptTemplate(input_variables=["context_with_metadata", "question"], template=template)



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

from collections import defaultdict

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
    print("\nTop Candidates:")
    for i, ((name, email), _) in enumerate(ranked[:top_k], 1):
        print(f"\n ## Candidate {i}: {name} ({email})")
        print("Relevant Info:")
        for chunk in candidate_chunks[(name, email)][:2]:
            print("•", chunk[:200].replace("\n", " "), "...")

def summarize_candidate_cv(candidate_name: str, vectorstore, llm_model="llama3.2"):
    llm = ChatOllama(model=llm_model)
    all_docs = vectorstore.get()
    matched_docs = [
        doc for doc in all_docs["documents"]
        if doc and candidate_name.lower() in doc.lower()
    ]

    if not matched_docs:
        matched_docs = [
            doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
            if meta.get("candidate_name", "").lower() == candidate_name.lower()
        ]

    if not matched_docs:
        print(f"❌ No documents found for candidate: {candidate_name}")
        return

    from langchain_core.documents import Document
    documents = [Document(page_content=doc) for doc in matched_docs]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(documents)

    print(f"\n📄 Summary for {candidate_name}:\n")
    print(summary)

