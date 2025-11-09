# Build stage
FROM python:3.11 AS builder

# Install Poetry
RUN pip install poetry==2.1.3 poetry-plugin-export

# Copy poetry files
COPY pyproject.toml poetry.lock /app/

# Set work directory
WORKDIR /app

# Generate requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements.txt from builder
COPY --from=builder /app/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose port 8000
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]