import json
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --- NOVEDAD: Función reutilizable para procesar un archivo de reglamento ---
def process_regulation_file(file_path, source_name):
    """
    Carga un archivo JSON, lo procesa y devuelve una lista de chunks con sus metadatos.

    Args:
        file_path (str): La ruta al archivo JSON.
        source_name (str): El nombre legible del documento para los metadatos.

    Returns:
        list: Una lista de diccionarios, donde cada uno es un chunk.
    """
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
                chunk_text = f"Article {sub_article['subarticle_number']} ({article['regulation_name']}): {sub_article['subarticle_text']}"

                # Usamos el source_name que pasamos a la función
                chunk_metadata = {
                    "source": source_name,
                    "article_title": article['regulation_name'],
                    "article_number": sub_article['subarticle_number']
                }

                processed_chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    print(f"Se crearon {len(processed_chunks)} chunks de texto desde '{source_name}'.")
    return processed_chunks

# --- PASO 1: Cargar y preparar los datos de TODAS las fuentes ---
print("Paso 1: Cargando y preparando los datos de todas las fuentes...")

# --- NOVEDAD: Define aquí todas tus fuentes de datos ---
# Es un diccionario donde la clave es el nombre del archivo y el valor es el nombre legible.
# ¡Puedes añadir tantos como quieras!
sources_to_process = {
    "json/FORMULA ONE SPORTING REGULATIONS.json": "FORMULA ONE SPORTING REGULATIONS",
    "json/INTERNATIONAL SPORTING CODE.json": "INTERNATIONAL SPORTING CODE",
    "json/APPENDIX L CHAPTER III - DRIVERS EQUIPMENT.json": "APPENDIX L CHAPTER III - DRIVERS EQUIPMENT",
    "json/APPENDIX L CHAPTER IV - CODE OF DRIVING CONDUCT ON CIRCUITS.json": "APPENDIX L CHAPTER IV - CODE OF DRIVING CONDUCT ON CIRCUITS"
}

all_chunks = []
for file_path, source_name in sources_to_process.items():
    chunks_from_source = process_regulation_file(file_path, source_name)
    all_chunks.extend(chunks_from_source) # Usamos .extend() para añadir los chunks a la lista principal

print(f"\nSe crearon {len(all_chunks)} chunks de texto en total de {len(sources_to_process)} fuentes.")

# --- EL RESTO DEL SCRIPT ES EXACTAMENTE IGUAL ---

# --- PASO 2: Convertir Texto a Vectores (Crear Embeddings) ---
print("\nPaso 2: Creando embeddings para cada chunk...")

model = SentenceTransformer('all-MiniLM-L6-v2')
texts_to_embed = [chunk['text'] for chunk in all_chunks]
embeddings = model.encode(texts_to_embed, show_progress_bar=True)

print(f"Embeddings creados. La forma del array es: {embeddings.shape}")

# --- PASO 3: Almacenar en una Base de Datos de Vectores ---
print("\nPaso 3: Creando y poblando la base de datos de vectores...")

db_path = "f1_regulations_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

client = chromadb.PersistentClient(path=db_path)
collection_name = "f1_regulations"
# Si la colección ya existe, la borramos para reconstruirla con los datos actualizados
# Esto es importante para evitar duplicados si ejecutas el script varias veces
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
    print(f"Colección '{collection_name}' existente eliminada para ser reconstruida.")

collection = client.create_collection(name=collection_name)

chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]

collection.add(
    embeddings=embeddings,
    documents=texts_to_embed,
    metadatas=[chunk['metadata'] for chunk in all_chunks],
    ids=chunk_ids
)

print(f"\n¡Éxito! Tu base de datos de vectores ha sido creada en la carpeta '{db_path}'.")
print(f"La colección '{collection_name}' ahora contiene {collection.count()} documentos de todas las fuentes.")