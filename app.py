"""
Streamlit front end for the job posting analyzer agent (agent.py).

Wraps the same agent.analyze() function the CLI (agent.py's run()) uses --
this file only handles input/output, all the agent logic lives in agent.py
and tools.py unchanged.
"""

import os

import streamlit as st

# Bridge Streamlit secrets into os.environ before importing agent/models, so
# LLM_PROVIDER and API keys are available the same way they are when set via
# a local .env file. This is what makes the same code work unmodified on
# Streamlit Community Cloud, Cloud Run, or anywhere else.
if hasattr(st, "secrets"):
    for key in ("LLM_PROVIDER", "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_MODEL"):
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

from agent import analyze  # noqa: E402  (import after secrets bridge, on purpose)

st.set_page_config(page_title="RoleFit", page_icon="\U0001F3AF", layout="centered")

st.title("RoleFit")
st.caption(
    "Paste a job posting and a resume. The agent extracts the posting's "
    "requirements, scores how well the resume covers them, and writes a "
    "gap analysis -- including an overall match score."
)

provider = os.environ.get("LLM_PROVIDER", "ollama")
st.caption(f"Model provider: `{provider}`")

col1, col2 = st.columns(2)
with col1:
    job_input = st.text_area(
        "Job posting",
        height=300,
        placeholder="Paste the job posting text here (or a URL to one)...",
    )
with col2:
    resume_input = st.text_area(
        "Resume",
        height=300,
        placeholder="Paste your resume text here...",
    )

run_clicked = st.button("Analyze fit", type="primary", use_container_width=True)

if run_clicked:
    if not job_input.strip() or not resume_input.strip():
        st.warning("Paste both a job posting and a resume before running the analysis.")
    else:
        with st.spinner("Running agent (this can take a minute, especially on a local model)..."):
            try:
                answer = analyze(job_input.strip(), resume_input.strip())
                st.markdown("---")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
