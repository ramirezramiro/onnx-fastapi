# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs that Pillow/SSL need
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only minimal runtime deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code (and your pre-exported model)
COPY app /app/app

# Drop privileges
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
