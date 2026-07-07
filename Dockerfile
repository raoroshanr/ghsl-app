# Cloud Run builds this into a container and runs it.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Cloud Run tells us which port to listen on via $PORT (usually 8080).
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
