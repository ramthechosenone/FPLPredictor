FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/

EXPOSE 8080

CMD ["uvicorn", "src.fpl.api:app", "--host", "0.0.0.0", "--port", "8080"]
