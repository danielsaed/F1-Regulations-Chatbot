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
import requests

# --- Importaciones para los Clientes de IA ---
import chromadb
from openai import AzureOpenAI
from fastapi.responses import StreamingResponse

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
    NORMAL_DEPLOYMENT_NAME = "gpt-5-nano"
    print("Cliente de Azure OpenAI configurado correctamente.")
except Exception as e:
    print(f"ERROR: No se pudo configurar el cliente de Azure OpenAI. Revisa tus variables de entorno. Error: {e}")
    exit()


# --- NUEVA Configuración para el Chat con AI Foundry ---
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY")
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")
# ¡IMPORTANTE! Debes especificar el nombre del modelo que desplegaste en Foundry
# Puede ser 'Llama-3-8B-chat', 'Mistral-Large', etc.
FOUNDRY_MODEL_NAME = "grok-3-mini" # <-- REEMPLAZA ESTO CON TU MODELO



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
            model=NORMAL_DEPLOYMENT_NAME, 
            messages=[{"role": "user", "content": rephrasing_prompt}]  # Queremos 0 creatividad, solo precisión.
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
    You are an expert in Formula 1 and search systems. Your task is to generate 1 alternative versions of the given user question.
    The goal is to improve the retrieval of relevant documents from a vector database that contains the official F1 regulations.
    The alternative questions should cover related concepts and use technical synonyms.

    Provide only the alternative questions, each on a new line. Do not number them.

    Original question: "{user_query}"

    Alternative questions:
    """
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=NORMAL_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": generation_prompt}],
            
            n=1
        )
        generated_text = response.choices[0].message.content
        alternative_queries = [q.strip() for q in generated_text.strip().split('\n') if q.strip()]
        return [user_query] + alternative_queries
    except Exception as e:
        print(f"Error al generar preguntas alternativas: {e}")
        return [user_query]

def find_relevant_chunks(user_query: str, top_k: int = 3) -> dict:
    # ¡CAMBIO CLAVE! Usamos Azure OpenAI para crear el embedding.
    query_embedding_result = azure_openai_client.embeddings.create(input=[user_query], model=EMBEDDING_DEPLOYMENT_NAME)
    query_embedding = query_embedding_result.data[0].embedding
    
    # La búsqueda en ChromaDB sigue siendo la misma.
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

def chunks_to_article(results_metadata: List[Dict]) -> List[str]:
    # Recupera el texto completo de los artículos "padre"
    lst_articles_titles = [f"{i['source']}-{i['article_title']}-{i['article_number']}" for i in results_metadata]

    #find duplicates to just give context of 2 or more finds on the articles
    seen = set()
    duplicates = set()

    for item in lst_articles_titles:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    lst_context = []
    dic_full_articles = {}
    for x in list(duplicates):
        x_temp = x.split("-")
        lst_context.append("Source = "+x_temp[0]+", Article = " + x_temp[2]+") " + x_temp[1] + " : "+dic_database.get(x_temp[0]+"-"+x_temp[1], ""))
        dic_full_articles["Source = "+x_temp[0]+", Article = " + x_temp[2]+") " + x_temp[1]] = lst_context[-1].split(":",1)


    return lst_context

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
    """
    Obtiene la respuesta del LLM usando el endpoint de Azure AI Foundry.
    """
    if not FOUNDRY_API_KEY or not FOUNDRY_ENDPOINT:
        raise HTTPException(status_code=500, detail="Azure AI Foundry endpoint or key not configured.")

    # Preparamos los encabezados para la autenticación
    headers = {
        "Authorization": f"Bearer {FOUNDRY_API_KEY}",
        "Content-Type": "application/json"
    }

    # Preparamos el cuerpo de la petición. Imita la estructura de OpenAI.
    payload = {
        "model": FOUNDRY_MODEL_NAME, # El nombre del modelo que desplegaste
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in F1 rules."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 4000
    }

    try:
        # Hacemos la llamada HTTP POST a la API
        response = requests.post(FOUNDRY_ENDPOINT, headers=headers, json=payload)
        
        # Verificamos si la petición fue exitosa
        response.raise_for_status() 
        
        # Extraemos la respuesta del JSON
        data = response.json()
        return data['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API de Foundry: {e}")
        # Reenviamos el error del servidor si es posible
        if e.response is not None:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to the Foundry endpoint.")
    except (KeyError, IndexError) as e:
        print(f"Error procesando la respuesta de Foundry: {e}")
        raise HTTPException(status_code=500, detail="Invalid response structure from the Foundry endpoint.")


async def get_llm_response_stream(prompt: str):
    """
    Obtiene la respuesta del LLM como un stream, acumulando tokens para mejor formato.
    """
    if not FOUNDRY_API_KEY or not FOUNDRY_ENDPOINT:
        yield "data: Azure AI Foundry endpoint or key not configured.\n\n"
        yield "data: [DONE]\n\n"
        return

    headers = {
        "Authorization": f"Bearer {FOUNDRY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": FOUNDRY_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in F1 rules."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 4000,
        "stream": True
    }
    
    try:
        buffer = ""
        
        with requests.post(FOUNDRY_ENDPOINT, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str.startswith('data: '):
                        if line_str.strip() == 'data: [DONE]':
                            # Enviar cualquier contenido restante en el buffer
                            if buffer.strip():
                                yield f"data: {buffer}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            json_str = line_str[6:]
                            data = json.loads(json_str)

                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    buffer += content
                                    
                                    # Enviar cuando tengamos un chunk significativo
                                    if should_send_chunk(buffer):
                                        yield f"data: {buffer}\n\n"
                                        buffer = ""
                                        
                        except json.JSONDecodeError:
                            continue
            
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        print(f"Error durante el streaming: {e}")
        yield f"data: Error during streaming: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

def should_send_chunk(buffer: str) -> bool:
    """
    Determina si debemos enviar el chunk actual basado en delimitadores naturales.
    """
    # Enviar si termina en punto, punto y coma, dos puntos, o salto de línea
    if buffer.endswith(('.', ';', ':', '\n', '!', '?')):
        return True
    
    # Enviar si termina un elemento de lista markdown
    if buffer.strip().endswith(')') and any(buffer.strip().startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']):
        return True
    
    # Enviar si el buffer es muy largo (fallback)
    if len(buffer) > 100:
        return True
        
    return False

# ==============================================================================
# 5. EL ENDPOINT PRINCIPAL DE LA API
# ==============================================================================

@app.post("/chat")  # Removemos response_model ya que es streaming
async def chat_endpoint(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Recibe una pregunta y el historial, y devuelve una respuesta con streaming.
    """
    try:
        # Valida la entrada
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # 1. Re-escribir la pregunta con el historial
        standalone_query = rephrase_query_with_history(request.query, request.history)
        print(f"Standalone query: {standalone_query}")
        
        # 2. Expandir la pregunta
        all_queries = generate_alternative_queries(standalone_query)
        print(f"Generated queries: {all_queries}")
        
        # 3. Buscar chunks relevantes
        all_results_metadata = []
        for q in all_queries:
            results = find_relevant_chunks(q)
            all_results_metadata.extend(results["metadatas"][0])
        
        # 4. Eliminar duplicados y obtener el contexto completo
        unique_metadata_tuples = set(tuple(d.items()) for d in all_results_metadata)
        unique_metadata = [dict(t) for t in unique_metadata_tuples]
        
        if not unique_metadata:
            # Si no hay contexto, devolvemos streaming con mensaje de error
            async def error_stream():
                yield "data: I couldn't find specific information about that topic in the F1 regulations. Could you try rephrasing your question?\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(error_stream(), media_type="text/plain")
        
        full_context_articles = chunks_to_article(all_results_metadata)

        print(full_context_articles)
        print(f"Retrieved {len(full_context_articles)} articles")
        
        # 5. Construir el prompt
        prompt = build_prompt(standalone_query, full_context_articles)
        
        # 6. Devolver respuesta streaming
        return StreamingResponse(get_llm_response_stream(prompt), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en el endpoint de chat: {str(e)}")
        
        # Stream de error
        async def error_stream():
            yield "data: Sorry, I encountered an internal error while processing your request. Please try again.\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(error_stream(), media_type="text/plain")

# ==============================================================================
# 6. (Opcional) Un endpoint de "salud" para verificar que la API está viva
# ==============================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok"}