FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the dependencies declared by this project.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the pipeline source and its supporting files.
COPY register_ctranslate2_model.py .
COPY services/ ./services/
COPY scripts/ ./scripts/
COPY mock_data/ ./mock_data/
COPY docs/ ./docs/
COPY README.md ./

# The model directory and evaluation data should be mounted/provided at runtime,
# not baked into the image.
ENTRYPOINT ["python", "register_ctranslate2_model.py"]
