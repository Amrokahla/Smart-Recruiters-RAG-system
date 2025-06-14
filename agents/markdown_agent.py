from crewai import Agent

# === Configurable constants ===
REQUIRED_FIELDS = ['Name', 'Email']
SECTION_ORDER = [
    "Personal Information", "Profile Overview", "Experience", "Education",
    "Projects", "Skills", "Courses", "Certificates", "Languages"
]

# === Markdown format example ===
MARKDOWN_TEMPLATE = """
# Personal Information
- **Name:** Omar Ayman
- **Email:** Omar.ayman@example.com
- **Phone:** +0201063998916
- **Links:** [LinkedIn](https://linkedin.com/in/johndoe), [GitHub](https://github.com/johndoe)
- **Military Service:** Optional
- **Languages:** Arabic, English


# Profile Overview
Brief paragraph summary of the candidate’s professional background.

# Experience
### Senior Software Engineer — ABC Corp (Jan 2020 – Present)
- 5+ years of experience
- Developed scalable web applications using React and Node.js
- Led a team of 4 developers

# Education
### B.Sc. in Computer Science — XYZ University (2018)
- Specialized in Software Engineering
- Graduated with Honors

# Projects
### Smart Recruiter (2024)
- Developed an AI-powered CV parser

# Skills
- Python, JavaScript, React, SQL
- Machine Learning, Data Analysis, Cloud Computing
- Git, Docker

# Courses
- **Coursera:** Machine Learning — Basics of supervised learning


# Certificates
- **AWS Certified Developer** — Amazon (link

# Languages
- English — Fluent
- Arabic — Native

"""

# === Agent Factory ===
def reformat_cv_agent(model):
    """
    Returns a CrewAI agent that formats unstructured CV text into
    clean, structured Markdown based on a predefined template.

    Args:
        model: A CrewAI-compatible model provider (e.g., OpenAI or HuggingFace wrapper).

    Returns:
        crewai.Agent object
    """
    return Agent(
        role="CV and Resume Formatter",
        goal="Convert raw CV text into a clean, structured Markdown format suitable for human reading and analysis.",
        backstory=(
            "You are a professional CV analyst and formatter. Your job is to read raw, unstructured resumes "
            "and reformat them into a standardized structure with clear headings and fields. You understand "
            "that CVs vary in layout and completeness, so you produce flexible, accurate outputs while "
            "ignoring missing or irrelevant fields."
        ),
        instructions=(
            f"Extract and structure the CV into the following Markdown format. Each main section should be a top-level heading. "
            f"Use consistent formatting with bold field labels. "
            f"**The following fields are required under the 'Personal Information' section: {', '.join(REQUIRED_FIELDS)}.**\n\n"
            "If any other field is missing, it can be omitted.\n\n"
            "⚠️ Output must be clean, properly formatted Markdown with no extra commentary, JSON, or explanation.\n\n"
            "**Markdown Structure:**\n\n"
            f"{MARKDOWN_TEMPLATE}\n\n"
            "⚠️ Repeat: Output only the final formatted Markdown document—no extra text or JSON."
        ),
        llm=model.get_llm(),
        verbose=False
    )
