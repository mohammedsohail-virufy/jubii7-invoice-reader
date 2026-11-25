# Small, non-root image for the API and dashboard.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JUBII7_ROOT=/app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
COPY models ./models
COPY reports ./reports
COPY docs ./docs

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "jubii7.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
