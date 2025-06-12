from crewai import Agent, Task

def reformat_cv_agent(model):
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
            "Extract and structure the CV into the following Markdown format. Each main section should be a top-level heading. "
            "Use consistent formatting with bold field labels. **The 'Name' and 'Email' fields must always be included under the 'Personal Information' section.** "
            "The first header will always be 'Personal Information'.\n\n"
            "If any other field is missing, it can be omitted.\n\n"

            "**Markdown Structure:**\n\n"
            "# Personal Information\n"
            "- **Name:** John Doe\n"
            "- **Email:** john.doe@example.com\n"
            "- **Phone:** +1 234 567 890\n"
            "- **Links:** [LinkedIn](https://linkedin.com/in/johndoe), [GitHub](https://github.com/johndoe)\n"
            "- **Military Service:** Optional\n\n"
            "# Profile Overview\n"
            "Brief paragraph summary of the candidate’s professional background.\n\n"
            "# Experience\n"
            "### Senior Software Engineer — ABC Corp (Jan 2020 – Present)\n"
            "- 5+ years of experience\n"
            "- Developed scalable web applications using React and Node.js\n"
            "- Led a team of 4 developers\n\n"
            "# Education\n"
            "### B.Sc. in Computer Science — XYZ University (2018)\n"
            "- Specialized in Software Engineering\n"
            "- Graduated with Honors\n\n"
            "# Projects\n"
            "### Smart Recruiter (2024)\n"
            "- Developed an AI-powered CV parser\n\n"
            "# Skills\n"
            "- Python, JavaScript, React, SQL\n\n"
            "# Courses\n"
            "- **Coursera:** Machine Learning — Basics of supervised learning\n\n"
            "# Certificates\n"
            "- **AWS Certified Developer** — Amazon (link)\n\n"
            "# Languages\n"
            "- English — Fluent\n"
            "- Spanish — Intermediate\n\n"
            "Use clean, readable formatting. Output only the final formatted Markdown document—no extra commentary or JSON."
        ),
        llm=model.get_llm(),
        verbose=False
    )