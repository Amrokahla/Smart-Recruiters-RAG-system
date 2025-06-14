from crewai import Agent

def extract_cv_metadata_agent(model):
    """
    CrewAI Agent that extracts metadata from a Markdown-formatted CV.

    It outputs a JSON object with:
        - source_file: The name of the CV Markdown file.
        - name: Candidate’s full name.
        - email: Candidate’s email address.


    Args:
        model: A CrewAI-compatible model with a .get_llm() method.

    Returns:
        crewai.Agent
    """
    return Agent(
        role="CV Metadata Extractor",
        goal="Extract essential metadata from a structured Markdown CV into a clean JSON format.",
        backstory=(
            "You are a metadata extraction specialist. You analyze structured Markdown-formatted CVs "
            "and accurately extract key fields like candidate name, email, and file name for indexing and retrieval."
        ),
        instructions=(
            "You will be given the contents of a Markdown-formatted CV along with the file name. "
            "Your task is to extract the following metadata and return it as a JSON object:\n\n"
            "- source_file: The name of the file provided (include the .md extension).\n"
            "- name: The name listed under the 'Personal Information' section after '**Name:**'.\n"
            "- email: The email listed under the 'Personal Information' section after '**Email:**'.\n\n"
            "Assume that 'Name' and 'Email' always exist in the document under the 'Personal Information' section. "
            "Only return a single JSON object in this exact format:\n\n"
            "{\n"
            '  "source_file": "Omar-Ayman-Elgemaey-Resume.md",\n'
            '  "name": "Omar Ayman Elgemaey",\n'
            '  "email": "3omargmi@gmail.com"\n'
            "}\n\n"
            "⚠️ Output only the valid JSON — no extra explanation, Markdown, or commentary."
        ),
        llm=model.get_llm(),
        verbose=False
    )
