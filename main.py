# main.py
import os
import json
from dotenv import load_dotenv

# --- Importaciones para FastAPI y Pydantic ---
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
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

# --- NUEVO: Lógica de Seguridad de API Key ---
API_KEY_NAME = "X-API-Key" # El nombre estándar para el encabezado de la clave
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Esta es la función "portero". Se ejecutará antes que nada en cada petición.
async def get_api_key(api_key: str = Security(api_key_header)):
    """Valida la clave de API enviada en el encabezado de la petición."""
    # Compara la clave recibida con la que tienes guardada de forma segura en Azure
    if api_key == os.getenv("BACKEND_API_KEY"):
        return api_key
    else:
        # Si no coincide, detiene la petición inmediatamente con un error.
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

# ==============================================================================
# 4. FUNCIONES DE LÓGICA DEL CHATBOT (ADAPTADAS)
# ==============================================================================

def rephrase_query_with_history(user_query: str, chat_history: List[ChatMessage]) -> str:
    """
    Usa el historial para convertir una pregunta de seguimiento en una pregunta independiente.
    """
    if not chat_history:
        return user_query

    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

    # --- PROMPT MEJORADO ---
    rephrasing_prompt = f"""
    You are an AI assistant. Your task is to rephrase a follow-up question to be a standalone question, based on a provided chat history.
    The new question must be in English, self-contained, and fully understandable without the context of the chat history.
    If the follow-up question is already a standalone question, simply return it as is.

    ---
    CHAT HISTORY:
    {history_str}
    ---
    FOLLOW UP QUESTION: "{user_query}"
    ---
    STANDALONE QUESTION:
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            # Usa un modelo rápido y barato para esta tarea interna. gpt-3.5-turbo es perfecto.
            model=CHEAP_DEPLOYMENT_NAME, 
            messages=[{"role": "user", "content": rephrasing_prompt}],
            temperature=0  # Queremos 0 creatividad, solo precisión.
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al reescribir la pregunta: {e}")
        return user_query # Si falla, es más seguro devolver la pregunta original

def generate_alternative_queries(user_query: str) -> List[str]:
    """
    Usa un LLM para generar 2 versiones alternativas de la pregunta del usuario.
    """
    # --- PROMPT MEJORADO ---
    generation_prompt = f"""
    You are an expert in Formula 1 and search systems. Your task is to generate 2 alternative versions of the given user question.
    The goal is to improve the retrieval of relevant documents from a vector database that contains the official F1 regulations.
    The alternative questions should cover related concepts and use technical synonyms.

    Provide only the alternative questions, each on a new line. Do not number them.

    Original question: "{user_query}"

    Alternative questions:
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=CHEAP_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=0.5, # Un poco de creatividad para que piense en sinónimos.
            n=1
        )
        generated_text = response.choices[0].message.content
        alternative_queries = [q.strip() for q in generated_text.strip().split('\n') if q.strip()]
        return [user_query] + alternative_queries
    except Exception as e:
        print(f"Error al generar preguntas alternativas: {e}")
        return [user_query]

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
    """
    Construye el prompt final para el LLM, combinando la pregunta y el contexto.
    """
    context = "\n\n---\n\n".join(context_articles)

    # --- PROMPT MEJORADO Y ESTRUCTURADO ---
    prompt = f"""
    You are a world-class expert assistant on Formula 1 regulations, your name is "Reggie".
    Your task is to answer the user's question with utmost precision, based ONLY on the provided context from the official regulations.

    CONTEXT FROM THE OFFICIAL REGULATIONS:
    ---
    {context}
    ---

    USER'S QUESTION:
    "{user_query}"

    YOUR INSTRUCTIONS:
    1.  **Analyze the Context:** Read the provided context carefully to find the answer.
    2.  **Answer the Question:** Formulate a clear, structured, and easy-to-understand answer. Use bullet points or numbered lists if it improves clarity, especially for procedures or lists of rules.
    3.  **Grounding is CRITICAL:** Your answer MUST be based exclusively on the information within the provided CONTEXT. Do not use any external knowledge.
    4.  **Cite Your Sources:** After providing a piece of information, you MUST cite the specific article number it came from. Use the format `[Source: DOCUMENT NAME - Article X.Y]`.
    5.  **Handle Missing Information:** If the answer cannot be found in the provided context, you MUST respond with the exact phrase: "Sorry, I could not find a definitive answer for this in the official documents provided. I work best with especific questions, please try again!"

    ANSWER:
    """
    return prompt

def get_llm_response(prompt: str) -> str:
    # Usa el cliente de Azure OpenAI para la respuesta final
    response = azure_openai_client.chat.completions.create(
        model=NORMAL_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==============================================================================
# 5. EL ENDPOINT PRINCIPAL DE LA API
# ==============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, api_key: str = Depends(get_api_key)):
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