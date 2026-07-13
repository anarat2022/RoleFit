# Containerized version of the Streamlit UI (app.py). This is the image
# Cloud Run deploys -- see the README's "Deploying on Google Cloud Run"
# section. Also runs anywhere else Docker is supported.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the port to listen on via the PORT env var (usually
# 8080) rather than a fixed value, so the run command reads it at startup.
ENV PORT=8080
EXPOSE 8080

# LLM_PROVIDER and API keys are passed in at deploy time (gcloud run deploy
# --set-env-vars, or the hosting platform's environment/secrets config),
# never baked into the image.
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
