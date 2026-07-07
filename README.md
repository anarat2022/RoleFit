# Job Posting Analyzer Agent

A tool-using agent, built with LangChain, that compares a job posting against a resume and produces an honest gap analysis: what matches, what's missing, and concrete suggestions for each gap.

## What makes this an "agent" and not just an API call

A single LLM call that reads a prompt and returns text isn't an agent. This is: the model is given three tools and a goal, and it decides for itself which tools to call, in what order, based on what the user actually gave it (a URL vs. pasted text, a file path vs. pasted resume text) and what it learns from each tool's result. That decision loop is built on LangChain's `create_agent()`, which wraps a LangGraph state machine: model turn -> tool call -> tool result fed back to the model -> model turn again -> ... until it produces a final answer instead of another tool call.

## Tools

- **`fetch_job_posting(url)`** -- fetches a job posting from a URL. Best-effort only: many job boards render with JavaScript or block scrapers, so this fails a lot in practice, and is designed to fail *usefully* -- it tells the agent (and the agent tells the user) to paste the text directly instead of silently returning garbage.
- **`read_resume_file(path)`** -- reads a local `.txt` resume file.
- **`score_skill_match(requirements, resume_text)`** -- deterministic keyword-overlap scoring. This one's the interesting design choice: it would be easy to just ask the LLM "does this resume cover requirement X?" for each requirement, but LLMs are inconsistent at that kind of literal counting/matching task. Doing it in plain Python instead makes the score reproducible and testable -- a good example of the general rule that agents should reach for a tool when a task needs to be reliable, not just when it needs an external API.

The requirement *extraction* itself (turning a wall of job posting text into a clean list of concrete requirements) is left to the model's own reasoning rather than a tool, since that's exactly the kind of unstructured-language task LLMs are good at and deterministic code isn't.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Default provider is Ollama (fully local, free, no key -- see the RAG project's README for install steps: `ollama pull llama3.2`, make sure the Ollama app is running). Set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` in `.env` to use a hosted model instead.

## Usage

```bash
python agent.py
```

It'll prompt you to paste a job posting (or a URL) and then your resume (text or a file path), each followed by pressing Enter twice. Try it first with the included samples:

```bash
python agent.py
# paste the contents of sample_data/sample_job_posting.txt
# then paste sample_data/sample_resume.txt, or type: sample_data/sample_resume.txt
```

## Example output

```
Match: 7/8 requirements (88%)

Matched:
  - LLM API experience (OpenAI, Anthropic, or Gemini)
  - RAG pipeline experience
  - vector database experience
  - agent framework experience (LangChain, LangGraph, or CrewAI)
  - Python
  - SQL
  - cloud platform experience (AWS, Azure, or GCP)

Missing (not found in resume text):
  - Kubernetes

Gap analysis: Strong fit overall -- 7 of 8 concrete requirements show up directly in your
resume, including the two most emphasized ones (LLM APIs and RAG pipelines). The one gap is
Kubernetes/container orchestration, listed as "nice to have" rather than required, so it's
not a blocker for applying. If you have any exposure to Docker or deployment pipelines, it's
worth a short resume line -- otherwise, this is an honest gap rather than something to word
around.
```
