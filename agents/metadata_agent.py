from crewai import Agent

def extract_cv_metadata_agent(model):
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
            "- markdown_file_name: The name of the file provided (include the .md extension).\n"
            "- candidate_name: The name listed under the 'Personal Information' section after '**Name:**'.\n"
            "- candidate_email: The email listed under the 'Personal Information' section after '**Email:**'.\n\n"
            "Assume that 'Name' and 'Email' always exist in the document under the 'Personal Information' section. "
            "Only return a single JSON object like this:\n\n"
            "{\n"
            '  "source_file": "john_doe_cv.md",\n'
            '  "name": "John Doe",\n'
            '  "email": "john.doe@example.com"\n'
            "}\n\n"
            "Return only valid JSON. Do not return any extra explanation or text."
        ),
        llm=model.get_llm(),
        verbose=False
    )
