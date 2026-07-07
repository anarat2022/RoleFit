"""
Tools for the job-posting-analyzer agent.

Three tools, each doing something the LLM itself either can't do (fetch a URL,
read a file) or shouldn't be trusted to do reliably (scoring skill overlap --
deterministic string matching in code beats asking an LLM to "count" matches).
That distinction is the actual design lesson here: agents should reach for a
tool when a task needs external I/O or needs to be reliable/reproducible, and
otherwise just let the model reason directly.
"""

import os
import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def fetch_job_posting(url: str) -> str:
    """Fetch a job posting from a URL and return its visible text content.

    Best-effort only: many job boards (LinkedIn, Indeed, etc.) render content
    with JavaScript or block scrapers, so this will fail on a lot of real
    postings. If it fails, tell the user to paste the job description text
    directly into the conversation instead of relying on this tool.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; job-analyzer-agent/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return (
            f"Could not fetch '{url}': {exc}. "
            "This page may block automated requests or require JavaScript. "
            "Ask the user to paste the job description text directly instead."
        )

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) < 200:
        return (
            f"Fetched '{url}' but got very little usable text ({len(cleaned)} chars) -- "
            "this page likely renders its content with JavaScript, which a plain HTTP "
            "request can't see. Ask the user to paste the job description text directly."
        )
    return cleaned[:8000]  # cap to keep the agent's context window reasonable


@tool
def read_resume_file(path: str) -> str:
    """Read a plain-text resume file from disk and return its contents.

    Only supports .txt files. If the user's resume is a PDF or DOCX, ask them
    to paste the text directly instead, or export it to .txt first.
    """
    if not os.path.exists(path):
        return f"No file found at '{path}'. Ask the user for the correct path, or have them paste their resume text directly."
    if not path.lower().endswith(".txt"):
        return f"'{path}' isn't a .txt file. Ask the user to paste their resume text directly, or export it as plain text first."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _normalize(text: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#/-]*", text.lower())
    return {w for w in words if len(w) > 2}


@tool
def score_skill_match(requirements: list[str], resume_text: str) -> str:
    """Score how well a resume covers a list of required skills/keywords.

    Deterministic keyword-overlap check (not an LLM judgment call) -- for
    each requirement, checks whether any of its significant words appear in
    the resume text. Returns matched requirements, missing requirements, and
    a match percentage. This is intentionally simple and literal: it will
    have false negatives if the resume phrases something differently than
    the job posting does, which is worth flagging to the user in the final
    analysis rather than silently trusting the score.
    """
    resume_words = _normalize(resume_text)

    matched = []
    missing = []
    for req in requirements:
        req_words = _normalize(req)
        if req_words & resume_words:
            matched.append(req)
        else:
            missing.append(req)

    total = len(requirements)
    match_pct = (len(matched) / total * 100) if total else 0.0

    matched_lines = [f"  - {m}" for m in matched] if matched else ["  (none)"]
    missing_lines = [f"  - {m}" for m in missing] if missing else ["  (none)"]

    lines = [
        f"Match: {len(matched)}/{total} requirements ({match_pct:.0f}%)",
        "",
        "Matched:",
        *matched_lines,
        "",
        "Missing (not found in resume text):",
        *missing_lines,
    ]
    return "\n".join(lines)


ALL_TOOLS = [fetch_job_posting, read_resume_file, score_skill_match]
