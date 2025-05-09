# ✅ 1. Use Python official image
FROM python:3.12-slim

# ✅ 2. Set working directory
WORKDIR /app

# ✅ 3. Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# ✅ 4. Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# ✅ 5. Copy project files
COPY . .

# ✅ 6. Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# ✅ 7. Expose API port
EXPOSE 8000

# ✅ 8. Run FastAPI server
CMD ["uvicorn", "indian_law_rag_chatbot.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
