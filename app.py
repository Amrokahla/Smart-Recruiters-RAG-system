import gradio as gr
from main import prepare_data, intialize_chromadb, load_vectorstore, build_rag_chain, process_uploaded_cv
from rag.rag import find_top_candidates, summarize_candidate_cv
from chromadb import Client
from chromadb.config import Settings
import os
import time

client_settings = Settings(persist_directory="chroma_store")
shared_client = Client(client_settings)

prepare_data()
intialize_chromadb()
vectorstore = load_vectorstore("chroma_store", shared_client)
qa_chain = build_rag_chain(vectorstore)

def ask_question(question, progress=gr.Progress(track_tqdm=True)):
    result = qa_chain.invoke(question)
    return result["result"]

def match_candidates(job_description, progress=gr.Progress(track_tqdm=True)):
    results = find_top_candidates(job_description, vectorstore)
    if not results:
        return "❌ No suitable candidates found for this job description."

    html = """
    <table style='width:100%; border-collapse: collapse;' border='1'>
        <tr style='background-color: #f2f2f2;'>
            <th style='padding: 8px;'>#</th>
            <th style='padding: 8px;'>Name</th>
            <th style='padding: 8px;'>Email</th>
            <th style='padding: 8px;'>Phone</th>
        </tr>
    """

    max_chunks = max(len(chunks) for _, chunks in results[:3]) or 1

    for i, ((name, email), chunks) in enumerate(results[:3], 1):
        phone = "Not found"
        for chunk in chunks:
            if "**Phone:**" in chunk:
                phone = chunk.split("**Phone:**")[-1].split("\n")[0].strip()

        #match_score = int((len(chunks) / max_chunks) * 100)

        html += f"""
        <tr>
            <td style='padding: 8px;'>{i}</td>
            <td style='padding: 8px;'>{name}</td>
            <td style='padding: 8px;'>{email}</td>
            <td style='padding: 8px;'>{phone}</td>
        </tr>
        """

    html += "</table>"
    return html

def summarize_cv(candidate_name, progress=gr.Progress(track_tqdm=True)):
    summary = summarize_candidate_cv(candidate_name, vectorstore)
    if not summary:
        return "❌ Candidate not found."
    return summary["summary"]

def get_all_candidate_names():
    try:
        collection = shared_client.get_collection("applicants")
        data = collection.get()
        names = list({meta.get("candidate_name") for meta in data["metadatas"] if meta.get("candidate_name")})
        return sorted(names)
    except Exception:
        return []

def upload_and_process_cvs(files, progress=gr.Progress(track_tqdm=True)):
    if not files:
        return "No files uploaded.", gr.update(choices=[])

    names = []
    for file in progress.tqdm(files, desc="Processing CVs"):
        name, _, _ = process_uploaded_cv(file.name, vectorstore)
        names.append(name)
        time.sleep(1)

    return "✅ CVs uploaded and analyzed successfully.", gr.update(choices=get_all_candidate_names())

with gr.Blocks(title="Smart Recruiter", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    <h1 style='text-align: center; color: #4a90e2;'> Smart Recruiter</h1>
    <p style='text-align: center;'>Upload CVs </p>
    """)

    file_input = gr.File(label="Upload CV(s) (PDF)", file_types=[".pdf"], file_count="multiple")
    upload_btn = gr.Button("Upload and Analyze", variant="primary")
    upload_status = gr.Textbox(visible=False)
    name_dropdown = gr.Dropdown(choices=[], label="Select a candidate")

    upload_btn.click(upload_and_process_cvs, inputs=file_input, outputs=[upload_status, name_dropdown], show_progress=True)

    gr.Markdown("---")

    with gr.Tab("Question Answering"):
        gr.Markdown("<h3 style='color: #2c3e50;'>Ask questions about candidates</h3>")
        question_input = gr.Textbox(label="Your question")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        ask_btn = gr.Button("Ask", variant="primary")
        ask_btn.click(ask_question, inputs=question_input, outputs=answer_output, show_progress=True)

    with gr.Tab("Job Matching"):
        gr.Markdown("<h3 style='color: #2c3e50;'>Find top candidates for a job</h3>")
        job_desc_input = gr.Textbox(label="Job description", lines=3)
        candidates_output = gr.HTML(label="Top candidates")
        match_btn = gr.Button("Find Candidates", variant="primary")
        match_btn.click(match_candidates, inputs=job_desc_input, outputs=candidates_output, show_progress=True)

    with gr.Tab("CV Summarization"):
        gr.Markdown("<h3 style='color: #2c3e50;'>Get a summary of a candidate's CV</h3>")
        summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
        summarize_btn = gr.Button("Summarize", variant="primary")
        summarize_btn.click(summarize_cv, inputs=name_dropdown, outputs=summary_output, show_progress=True)

if __name__ == "__main__":
    app.launch()
