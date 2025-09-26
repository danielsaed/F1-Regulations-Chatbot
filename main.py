# main.py
import os
import json
from dotenv import load_dotenv

# --- Importaciones para FastAPI y Pydantic ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# --- Importaciones para los Clientes de IA ---
import chromadb
from openai import AzureOpenAI

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE MODELOS (Se ejecuta UNA SOLA VEZ al iniciar la API)
# ==============================================================================

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración del Cliente de Azure OpenAI ---
# Se usará tanto para embeddings como para las respuestas del chat.
try:
    azure_openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    EMBEDDING_DEPLOYMENT_NAME = "f1-embeddings" 
    CHEAP_DEPLOYMENT_NAME = "gpt-35-turbo"
    NORMAL_DEPLOYMENT_NAME = "gpt-5-mini"
    print("Cliente de Azure OpenAI configurado correctamente.")
except Exception as e:
    print(f"ERROR: No se pudo configurar el cliente de Azure OpenAI. Revisa tus variables de entorno. Error: {e}")
    exit()

# --- Conexión con la Base de Datos de Vectores (ChromaDB) ---
# Esto es muy ligero y rápido, ya no carga un modelo en memoria.
try:
    client = chromadb.PersistentClient(path="./f1_regulations_db")
    collection = client.get_collection(name="f1_regulations_azure") # La colección con embeddings de Azure
    print(f"Conectado a ChromaDB. La colección contiene {collection.count()} documentos.")
except Exception as e:
    print(f"ERROR: No se pudo conectar a ChromaDB. Asegúrate de que la carpeta 'f1_regulations_db' existe. Error: {e}")
    exit()

# --- Carga de Documentos "Padre" para el Contexto ---
# Creamos un diccionario para el acceso rápido a los textos completos de los artículos.
def load_parent_documents():
    db = {}
    sources_to_process = {
        "json/FORMULA ONE SPORTING REGULATIONS.json": "FORMULA ONE SPORTING REGULATIONS",
        "json/INTERNATIONAL SPORTING CODE.json": "INTERNATIONAL SPORTING CODE"
    }
    for file_path, source_name in sources_to_process.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for article in data:
                key = f"{source_name}-{article['regulation_name']}"
                full_text = "\n".join([sub['subarticle_text'] for sub in article['subarticles']])
                db[key] = full_text
    return db

dic_database = load_parent_documents()
print("Base de datos de documentos 'padre' cargada en memoria.")


# ==============================================================================
# 2. DEFINICIÓN DE LOS MODELOS DE DATOS (PYDANTIC)
# ==============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict] = []

# ==============================================================================
# 3. INICIALIZACIÓN DE LA APLICACIÓN FASTAPI
# ==============================================================================

app = FastAPI(
    title="F1 Regulations AI Assistant API",
    description="Una API para hacer preguntas sobre el reglamento de la F1 usando un sistema RAG.",
    version="1.0.0"
)

# Configuración de CORS (Cross-Origin Resource Sharing)
# Esto es crucial para permitir que tu frontend (en otro dominio) se comunique con esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, deberías restringirlo a la URL de tu frontend.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 4. FUNCIONES DE LÓGICA DEL CHATBOT (ADAPTADAS)
# ==============================================================================

def rephrase_query_with_history(user_query: str, chat_history: List[ChatMessage]) -> str:
    if not chat_history:
        return user_query
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    rephrasing_prompt = f"""Based on the chat history below, rephrase the "Follow Up Question" to be a self-contained, standalone question in English...""" # Tu prompt completo aquí
    
    response = azure_openai_client.chat.completions.create(
        model=CHEAP_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": rephrasing_prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def generate_alternative_queries(user_query: str) -> List[str]:
    # ... (Tu función completa aquí, usando azure_openai_client)
    generation_prompt = f"""You are a helpful AI assistant..."""
    response = azure_openai_client.chat.completions.create(model=CHEAP_DEPLOYMENT_NAME, messages=[{"role": "user", "content": generation_prompt}], temperature=0.5, n=1)
    generated_text = response.choices[0].message.content
    alternative_queries = [q.strip() for q in generated_text.strip().split('\n') if q.strip()]
    return [user_query] + alternative_queries

def find_relevant_chunks(user_query: str, top_k: int = 4) -> dict:
    # ¡CAMBIO CLAVE! Usamos Azure OpenAI para crear el embedding.
    query_embedding_result = azure_openai_client.embeddings.create(input=[user_query], model=EMBEDDING_DEPLOYMENT_NAME)
    query_embedding = query_embedding_result.data[0].embedding
    
    # La búsqueda en ChromaDB sigue siendo la misma.
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

def chunks_to_article(results_metadata: List[Dict]) -> List[str]:
    # Recupera el texto completo de los artículos "padre"
    lst_articles_titles = [f"{i['source']}-{i['article_title']}" for i in results_metadata]
    lst_articles_titles = list(set(lst_articles_titles))
    return [dic_database.get(x, "") for x in lst_articles_titles]

def build_prompt(user_query: str, context_articles: List[str]) -> str:
    # ... (Tu función completa aquí)
    context = "\n\n---\n\n".join(context_articles)
    return f"""You are an expert assistant for Formula 1 regulations... CONTEXT: {context} QUESTION: {user_query} ..."""

def get_llm_response(prompt: str) -> str:
    # Usa el cliente de Azure OpenAI para la respuesta final
    response = azure_openai_client.chat.completions.create(
        model=CHEAP_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==============================================================================
# 5. EL ENDPOINT PRINCIPAL DE LA API
# ==============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recibe una pregunta y el historial, y devuelve una respuesta con la lógica RAG completa.
    """
    try:
        # 1. Re-escribir la pregunta con el historial
        standalone_query = rephrase_query_with_history(request.query, request.history)
        print("1")
        
        # 2. Expandir la pregunta
        all_queries = generate_alternative_queries(standalone_query)
        print("2")
        
        # 3. Buscar chunks relevantes
        all_results_metadata = []
        for q in all_queries:
            results = find_relevant_chunks(q)
            all_results_metadata.extend(results["metadatas"][0])
        print("3")
        
        # 4. Eliminar duplicados y obtener el contexto completo
        unique_metadata_tuples = set(tuple(d.items()) for d in all_results_metadata)
        unique_metadata = [dict(t) for t in unique_metadata_tuples]
        
        full_context_articles = chunks_to_article(unique_metadata)
        print("4")
        
        # 5. Construir el prompt
        prompt = build_prompt(standalone_query, full_context_articles)
        print("5")
        
        # 6. Generar la respuesta
        answer = get_llm_response(prompt)
        
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate a response from the LLM.")
            
        # 7. Devolver la respuesta en el formato correcto
        return ChatResponse(answer=answer, sources=unique_metadata)
        
    except Exception as e:
        # Captura cualquier error inesperado y devuelve una respuesta de error clara.
        print(f"Error en el endpoint de chat: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# ==============================================================================
# 6. (Opcional) Un endpoint de "salud" para verificar que la API está viva
# ==============================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok"}