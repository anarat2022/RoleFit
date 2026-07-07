"""
Job Posting Analyzer -- a tool-using agent built with LangChain.

Give it a job posting (URL or pasted text) and a resume (file path or pasted
text), and it will: extract the concrete requirements from the posting, check
which ones show up in the resume using a deterministic keyword-match tool,
and write a gap analysis with suggested resume-bullet phrasing for anything
missing.

This uses LangChain's current (1.x) agent API: create_agent() builds a
tool-calling loop on top of LangGraph under the hood -- the model decides
which tool to call and when, calls it, sees the result, and decides whether
to call another tool or give a final answer. That loop (not a fixed script)
is what makes this an "agent" rather than a regular function call chain.

Usage:
    python agent.py
    (then paste a job posting and your resume path/text when prompted)
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from models import build_chat_model
from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a job application analysis agent. Your job: given a job \
posting and a candidate's resume, produce a clear, honest gap analysis.

You have three tools:
- fetch_job_posting: use this ONLY if the user gives you a URL instead of pasted text.
- read_resume_file: use this ONLY if the user gives you a file path instead of pasted resume text.
- score_skill_match: ALWAYS use this tool to check requirement coverage. First extract a \
list of concrete, specific requirements from the job posting yourself (skills, tools, \
years of experience, degree requirements, etc. -- not vague phrases), then pass that \
list plus the resume text to score_skill_match.

After you get the tool's matched/missing breakdown, write a final answer with EXACTLY this structure:

1. First line, always: "Overall Match Score: X%" -- copy this number directly from the \
score_skill_match tool's output. Never estimate, round differently, or invent your own \
number -- it must be the exact percentage the tool returned.
2. A short summary of fit (2-3 sentences) referencing the specific matched requirements.
3. The specific missing requirements, listed clearly.
4. For each missing requirement, one concrete suggestion: either a resume bullet the \
candidate could add if they have relevant-but-differently-worded experience, or an honest \
note that this is a real gap worth addressing before applying.

Be honest, not falsely encouraging. If the match is weak, say so plainly -- a low score is \
useful information, not a failure to soften. Never suggest padding a resume with skills the \
candidate doesn't actually have."""


def run():
    model = build_chat_model()
    agent = create_agent(model=model, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)

    print("Job Posting Analyzer (LangChain agent)")
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'ollama')}")
    print("-" * 60)
    print("Paste a job posting (or a URL to one), then press Enter twice:")
    job_input = _read_multiline()

    print("\nNow paste your resume text (or a path to a .txt file), then press Enter twice:")
    resume_input = _read_multiline()

    user_message = (
        f"Here is the job posting (URL or text):\n{job_input}\n\n"
        f"Here is my resume (file path or text):\n{resume_input}\n\n"
        "Please analyze my fit for this role."
    )

    print("\nRunning agent (this may take a minute, especially on a local model)...\n")
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    final_message = result["messages"][-1]
    print("=" * 60)
    print(final_message.content)


def _read_multiline() -> str:
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    run()
