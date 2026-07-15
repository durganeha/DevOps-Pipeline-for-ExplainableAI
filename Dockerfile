FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Train a baseline model at build time so the image is immediately usable
# (falls back to synthetic data automatically if data/raw/loan_data.csv
# hasn't been added yet — see src/data_loader.py)
RUN python src/train_model.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]