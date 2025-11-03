FROM python:3.11-slim

# Add this line so xgboost can load libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]

