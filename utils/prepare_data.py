import os
import fitz
import json
from crewai import Crew, Task
from model.model_config import load_config
from model.model import GeminiModel
from agents.markdown_agent import reformat_cv_agent
from agents.metadata_agent import extract_cv_metadata_agent
from tqdm import tqdm
import time

def read_text(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def clean_markdown_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if content.startswith('```markdown\n') and content.endswith('\n```'):
            cleaned_content = content[len('```markdown\n'):-len('\n```')]
        elif content.startswith('```markdown') and content.endswith('```'):
            cleaned_content = content[len('```markdown'):-len('```')]
        elif content.startswith('```\n') and content.endswith('\n```'):
            cleaned_content = content[len('```\n'):-len('\n```')]
        elif content.startswith('```') and content.endswith('```'):
            cleaned_content = content[len('```'):-len('```')]
        else:
            cleaned_content = content.strip()

        with open(file_path, 'w') as f:
            f.write(cleaned_content.strip())

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def cv2md(path):
    cv_text = read_text(path)
    config = load_config("model/config/config.yaml")
    model = GeminiModel(config)
    cv_agent = reformat_cv_agent(model)

    cv_format_task = Task(
    description=(
            "Take a raw, unstructured CV text and convert it into a clean, structured markdown object. "
            "Ensure the output follows a standardized format with clear section headings such as personal_info, "
            "profile_overview, experience, education, projects, skills, courses, certificates, and languages. "
            "Allow for missing fields and omit them gracefully in the markdown output."
            f"Here is the CV:\n\n{cv_text}"
            ),
    agent=cv_agent,
    expected_output="A well-structured markdown object containing cleaned and categorized CV data"
        )
    
    crew = Crew(agents= [cv_agent],
                tasks = [cv_format_task],
                verbose = False)
    result = crew.kickoff()
    
    file_name = path.split("\\")[-1].split('.')[0] + ".md"
    target_path = "data\markdowns"
    target_path = os.path.join(target_path,file_name)
    with open(target_path, 'w',encoding='utf-8') as f:
        f.write(str(result))
    clean_markdown_file(target_path)


def generate_markdowns(parent_path):
    cvs = os.listdir(os.path.join(parent_path,'pdf'))
    print("Processing CVs")
    for cv in tqdm(cvs):
        path = os.path.join(parent_path,'pdf',cv)
        cv2md(path)

def md2meta(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    markdown_file_name = os.path.basename(file_path)

    config = load_config("model/config/config.yaml")
    model = GeminiModel(config)
    metadata_agent = extract_cv_metadata_agent(model)

    metadata_task = Task(
        description=(
            f"You are given a markdown-formatted CV document and its file name.\n\n"
            f"Markdown File Name: {markdown_file_name}\n\n"
            f"Markdown Content:\n{markdown_content}"
        ),
        agent=metadata_agent,
        expected_output="A valid JSON object with fields: markdown_file_name, candidate_name, candidate_email"
    )

    crew = Crew(agents=[metadata_agent], tasks=[metadata_task], verbose=False)
    result = crew.kickoff()

    json_path = file_path.replace("markdowns", "metadata").replace(".md", ".json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding='utf-8') as f:
        f.write(str(result))
        

def clean_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if content.startswith('```json\n') and content.endswith('\n```'):
            content = content[len('```json\n'):-len('\n```')].strip()
        elif content.startswith('```json') and content.endswith('```'):
            content = content[len('```json'):-len('```')].strip()
        elif content.startswith('```\n') and content.endswith('\n```'):
            content = content[len('```\n'):-len('\n```')].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[len('```'):-len('```')].strip()

        lines = content.splitlines()
        while lines and not lines[0].lstrip().startswith('{'):
            lines.pop(0)
        content = "\n".join(lines).strip()

        parsed = json.loads(content)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, indent=2)

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while cleaning JSON: {e}")



def generate_metadata(parent_path):
    markdowns_path = os.path.join(parent_path, "markdowns")
    markdowns = [f for f in os.listdir(markdowns_path) if f.endswith(".md")]
    json_path = os.path.join(parent_path, "metadata")
    print("Processing Markdown files for metadata extraction")
    for md_file in tqdm(markdowns):
        md2meta(os.path.join(markdowns_path, md_file))
        json_file = md_file.replace(".md", ".json")
        clean_json_file(os.path.join(json_path, json_file))
        time.sleep(4)
