FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# создаём data-каталог до смены пользователя, чтобы приложение могло туда писать
RUN mkdir -p /app/data && chmod -R 777 /app/data

RUN useradd -m -u 1000 user
USER user

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/dashboard || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
