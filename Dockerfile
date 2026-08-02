FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/home/user/.cache/huggingface
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/app

COPY --chown=user requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN grep -viE '^[[:space:]]*torch([<>=!~].*)?$' requirements.txt > /tmp/requirements-docker.txt && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
