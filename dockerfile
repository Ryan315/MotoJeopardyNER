# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model and NLTK data
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('words')"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/subsets outputs/logs outputs/processed outputs/reports outputs/validation

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
CMD ["python", "scripts/main_pipeline.py"]