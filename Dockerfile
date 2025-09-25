# =========================================================================
# Etapa 1: Construcción de la Base y las Dependencias
# Usaremos una "multi-stage build" para mantener la imagen final más pequeña.
# =========================================================================
FROM python:3.10-slim as builder

# Establecemos el directorio de trabajo
WORKDIR /app

# Actualizamos pip e instalamos Gunicorn y Uvicorn
RUN pip install --upgrade pip
RUN pip install gunicorn uvicorn

# Copiamos solo el archivo de requerimientos para instalar las dependencias.
# Esto aprovecha el caché de Docker: si este archivo no cambia, Docker no
# volverá a instalar todo, haciendo las builds mucho más rápidas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =========================================================================
# Etapa 2: Creación de la Imagen Final de Producción
# =========================================================================
FROM python:3.10-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos las dependencias ya instaladas de la etapa anterior
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiamos todo el código de nuestra aplicación, incluyendo la base de datos de ChromaDB
# y el JSON de los reglamentos.
COPY . .

# (PASO CRÍTICO DE OPTIMIZACIÓN)
# Descargamos y cacheadamos el modelo de embeddings DURANTE la construcción.
# Esto evita la descarga de 500MB cada vez que el contenedor se inicia,
# haciendo que tu aplicación arranque en segundos en lugar de minutos.
# Asegúrate de que el nombre del modelo aquí coincida con el de tu código.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Exponemos el puerto en el que Gunicorn se ejecutará dentro del contenedor
EXPOSE 8000

# El comando final para ejecutar la aplicación en producción.
# Gunicorn es el servidor que maneja los workers (procesos).
# Uvicorn es el worker que ejecuta el código de FastAPI.
# -w 4: Inicia 4 "workers". Tu app puede manejar 4 peticiones simultáneamente.
# -k uvicorn.workers.UvicornWorker: Especifica que cada worker será de tipo Uvicorn.
# -b 0.0.0.0:8000: "Bind" (enlaza) el servidor a todas las interfaces de red en el puerto 8000.
# Esto es esencial para que Azure pueda comunicarse con tu aplicación.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]