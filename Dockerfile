FROM python:3.10-slim as builder
WORKDIR /app
RUN pip install --upgrade pip gunicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Etapa 2: Imagen final de producción
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiamos todo el código, incluyendo la DB de Chroma y los JSON
COPY . .

EXPOSE 8000

# El comando de inicio es el mismo
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]