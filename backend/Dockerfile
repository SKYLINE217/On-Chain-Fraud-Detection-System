FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# BUG-30 Note: Model checkpoints and processed data are NOT baked into the image.
# They are large (>1GB) and change frequently. Mount them as volumes at runtime:
#
#   docker run -v $(pwd)/checkpoints:/app/checkpoints \
#              -v $(pwd)/data/processed:/app/data/processed \
#              onchain-fraud-api
#
# Or via docker-compose with:
#   volumes:
#     - ./checkpoints:/app/checkpoints
#     - ./data/processed:/app/data/processed
#
# For CI/CD, pull checkpoints from artifact storage (S3, GCS, etc.)
# before building / running the container.

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
