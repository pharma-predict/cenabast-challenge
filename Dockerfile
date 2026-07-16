FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Entrenar el modelo durante el build (queda embebido como model.pkl en la imagen,
# evitando reentrenar en cada cold start de Cloud Run).
RUN python -m challenge.train

# Expose port 8080 (required by Cloud Run)
EXPOSE 8080

# Run the API with uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
