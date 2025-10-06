import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI # <-- CAMBIO: Importamos el cliente de Azure
import chromadb

# --- CONFIGURACIÓN: Carga las claves desde tu archivo .env ---
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# ¡IMPORTANTE! El nombre de tu despliegue de embeddings en Azure
EMBEDDING_DEPLOYMENT_NAME = "f1-embeddings" 

# --- TU LÓGICA ORIGINAL (CASI SIN CAMBIOS) ---

def process_regulation_file(file_path, source_name):
    # ... (Esta función es idéntica a la tuya, la copias y pegas aquí)
    print(f"Procesando el archivo: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            regulations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no se encontró. Saltando este archivo.")
        return []
    processed_chunks = []
    for article in regulations_data:
        if "subarticles" in article and article["subarticles"]:
            for sub_article in article['subarticles']:
                chunk_text = f"Source: {source_name}, Article {sub_article['subarticle_number']} ({article['regulation_name']}): {sub_article['subarticle_text']}"
                chunk_metadata = {
                    "source": source_name,
                    "article_title": article['regulation_name'],
                    "article_number": sub_article['subarticle_number']
                }
                processed_chunks.append({"text": chunk_text, "metadata": chunk_metadata})
    print(f"Se crearon {len(processed_chunks)} chunks de texto desde '{source_name}'.")
    return processed_chunks

print("Paso 1: Cargando y preparando los datos de todas las fuentes...")
"""sources_to_process = {
    "json/FORMULA ONE SPORTING REGULATIONS.json": "FORMULA ONE SPORTING REGULATIONS",
    "json/INTERNATIONAL SPORTING CODE.json": "INTERNATIONAL SPORTING CODE"
}"""
sources_to_process = {
    "json/FORMULA ONE SPORTING REGULATIONS.json": "FORMULA ONE SPORTING REGULATIONS",
    "json/INTERNATIONAL SPORTING CODE.json": "INTERNATIONAL SPORTING CODE",
    "json/APPENDIX L CHAPTER III DRIVERS EQUIPMENT.json": "APPENDIX L CHAPTER III DRIVERS EQUIPMENT",
    "json/APPENDIX L CHAPTER IV CODE OF DRIVING CONDUCT ON CIRCUITS.json": "APPENDIX L CHAPTER IV CODE OF DRIVING CONDUCT ON CIRCUITS"
}
all_chunks = []
for file_path, source_name in sources_to_process.items():
    chunks_from_source = process_regulation_file(file_path, source_name)
    all_chunks.extend(chunks_from_source)
print(f"\nSe crearon {len(all_chunks)} chunks de texto en total.")


# --- PASO 2: Crear Embeddings usando Azure OpenAI (EL CAMBIO CLAVE) ---
print("\nPaso 2: Creando embeddings para cada chunk usando Azure OpenAI...")

# Inicializamos el cliente de Azure OpenAI
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

texts_to_embed = [chunk['text'] for chunk in all_chunks]

# Hacemos la llamada a la API de Azure para generar los embeddings
embedding_result = azure_openai_client.embeddings.create(input=texts_to_embed, model=EMBEDDING_DEPLOYMENT_NAME)
embeddings = [item.embedding for item in embedding_result.data]

print(f"Embeddings creados vía Azure. Se generaron {len(embeddings)} vectores.")

# --- PASO 3: Almacenar en ChromaDB (SIN CAMBIOS) ---
print("\nPaso 3: Creando y poblando la base de datos de vectores en ChromaDB...")
# Esta parte es idéntica a tu script original. Funciona perfectamente.
db_path = "f1_regulations_db"
client = chromadb.PersistentClient(path=db_path)
collection_name = "f1_regulations_azure" # Le damos un nuevo nombre para no mezclar
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
collection = client.create_collection(name=collection_name)
chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
collection.add(
    embeddings=embeddings,
    documents=texts_to_embed,
    metadatas=[chunk['metadata'] for chunk in all_chunks],
    ids=chunk_ids
)
print(f"\n¡Éxito! Tu base de datos ChromaDB ha sido creada usando los embeddings de Azure.")