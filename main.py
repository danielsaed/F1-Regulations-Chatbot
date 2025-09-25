import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import openai

# --- 1. Importaciones para FastAPI ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- 2. Carga de Modelos y Datos (Se ejecuta UNA SOLA VEZ al iniciar la API) ---

# Carga las variables de entorno (tu API key) desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

print("Cargando el modelo de embeddings y la base de datos de vectores...")
# Carga el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
# Conecta con la base de datos de vectores
client = chromadb.PersistentClient(path="./f1_regulations_db")
collection = client.get_collection(name="f1_regulations")
print("¡Modelos y base de datos cargados con éxito!")

# --- (NUEVO Y CRÍTICO) Cargar los documentos "padre" para el contexto ---
# Tu función `chunks_to_article` necesita este diccionario para funcionar.
# Lo cargamos una vez al inicio para máxima eficiencia.
def load_parent_documents():
    import json
    # Cargamos tu JSON original
    with open('regulations.json', 'r', encoding='utf-8') as f:
        regulations_data = json.load(f)
    
    # Creamos el diccionario que mapea "Fuente-Título" al texto completo del artículo
    db = {}
    for article in regulations_data:
        print(article)
        # La clave será como "2025 F1 Sporting Regulations-SAFETY CAR"
        key = f"{article['subarticles'][0]['document']}-{article['regulation_name']}"
        
        # Juntamos todo el texto de los subartículos para tener el contexto completo
        full_text = "\n".join([sub['subarticle_text'] for sub in article['subarticles']])
        db[key] = full_text
    return db

dic_database = load_parent_documents()
print("Base de datos de documentos 'padre' cargada.")

# --- 3. Definición de los Modelos de Datos (Pydantic) ---
# Esto le dice a FastAPI cómo deben ser los datos de entrada y salida de tu API.

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict] # Para devolver las fuentes usadas

# --- 4. Inicializar la Aplicación FastAPI ---
app = FastAPI(
    title="F1 Regulations AI Assistant API",
    description="Una API para hacer preguntas sobre el reglamento de la F1",
    version="1.0.0"
)

# --- 5. Tus Funciones de Lógica (Copiadas de Colab) ---
# Todas tus funciones de lógica van aquí, sin cambios.
# (He corregido el pequeño bug en `chunks_to_article` para que use `dic_database`)

def rephrase_query_with_history(user_query, chat_history):
    """
    Usa el historial para convertir una pregunta de seguimiento en una pregunta independiente.
    """
    # Si no hay historial, simplemente devuelve la pregunta original.
    if not chat_history:
        return user_query

    try:
        # Formateamos el historial en un string simple
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    except:
        return user_query

    # Un prompt muy específico para la tarea de re-escribir
    rephrasing_prompt = f"""
    Based on the chat history below, rephrase the "Follow Up Question" to be a self-contained, standalone question in English.
    The new question must be understood without the chat history, always give more importance to the last interaction, and be careful, sometimes the user changes the topic so if some early parts of the history are not useful do not take it in count, remember that is a chat history interaction, treat it like that.

    CHAT HISTORY:
    {history_str}

    FOLLOW UP QUESTION: "{user_query}"

    STANDALONE QUESTION:
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Usamos un modelo rápido y barato para esta tarea interna
            messages=[{"role": "user", "content": rephrasing_prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al reescribir la pregunta: {e}")
        return user_query # Si falla, es más seguro devolver la pregunta original

def find_relevant_chunks(user_query, top_k=4):
    """
    Encuentra los chunks más relevantes para una pregunta en la base de datos de vectores.
    """
    query_embedding = model.encode(user_query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

def chunks_to_article(results_metadata):
    # (VERSIÓN CORREGIDA)
    lst_articles_titles = [f"{i['source']}-{i['article_title']}" for i in results_metadata]
    lst_articles_titles = list(set(lst_articles_titles))
    lst_full_articles_context = [dic_database.get(x, "") for x in lst_articles_titles]
    return lst_full_articles_context

def generate_alternative_queries(user_query):
    """
    Usa un LLM para generar 3 versiones alternativas de la pregunta del usuario.
    """
    try:
        # Este es un prompt diseñado para la tarea de reescribir preguntas
        generation_prompt = f"""
        You are a helpful AI assistant. Your task is to generate 2 alternative versions of the given user question in english.
        The goal is to improve the retrieval of relevant documents from a vector database that contains the F1 sporting regulations.
        Provide only the alternative questions, each on a new line. Do not number them, relation all the concepts with motorsports and Formula 1.

        Original question: {user_query}

        Alternative questions:
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": generation_prompt}
            ],
            temperature=0.5, # Un poco de creatividad para reescribir, pero no demasiada
            n=1 # Queremos una sola respuesta
        )

        generated_text = response.choices[0].message.content

        # Limpiamos el resultado y lo convertimos en una lista
        alternative_queries = [q.strip() for q in generated_text.strip().split('\n') if q.strip()]

        # Devolvemos la lista original + las nuevas para una búsqueda completa
        all_queries = [user_query] + alternative_queries
        return all_queries

    except Exception as e:
        print(f"Error al generar preguntas alternativas: {e}")
        # Si falla, simplemente devolvemos la pregunta original
        return [user_query]

def build_prompt(user_query, results):
    """
    Construye el prompt para el LLM, combinando la pregunta y el contexto.
    """
    #context_chunks = results['documents'][0]

    context = "\n\n---\n\n".join(results)

    prompt = f"""
    You are an expert assistant for Formula 1 regulations. Your task is to answer the user's question based ONLY on the provided context.
    Relation all the concepts with motorsports and Formula 1

    CONTEXT:
    {context}

    QUESTION:
    {user_query}

    INSTRUCTIONS:
    - Answer the question clearly and concise
    - Always order the response when explaining a regulation, example if you are explaining the podium procedure, start with the events that comes first, like this, - driver needs to do an interview and wear a cap after finish the race, - driver needs to go to the podium wait room after interviews, - driver must respect national athems, - drivers have to go to the conference room after podium
    - If the context does not contain the answer, you MUST say "Sorry I could not find an answer for this on the oficial documents, I work best with specifict questions, feel free to ask again! ." if necesary tranlate this text to the user language
    - Do not make up information or use external knowledge.
    - Cite the relevant article number(s) in your answer, for example: [ INTERNATIONAL SPORTING REGULATION - Article 55.1].

    ANSWER:
    """
    return prompt

def get_llm_response(prompt):
    """
    Obtiene la respuesta del LLM usando la API de OpenAI.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in F1 rules."},
                {"role": "user", "content": prompt},

            ],
            max_completion_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ocurrió un error con la API de OpenAI: {e}")
        return None 

# --- 6. El Endpoint de la API ---
# Este es el punto de entrada que tu frontend llamará.


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recibe una pregunta y el historial, y devuelve una respuesta con la lógica RAG.
    """
    # 1. Re-escribir la pregunta con el historial
    standalone_query = rephrase_query_with_history(request.query, request.history)
    
    # 2. Expandir la pregunta
    all_queries = generate_alternative_queries(standalone_query)
    
    # 3. Buscar chunks relevantes
    all_results_metadata = []
    for q in all_queries:
        results = find_relevant_chunks(q)
        all_results_metadata.extend(results["metadatas"][0])
    
    # 4. Eliminar duplicados y obtener el contexto completo
    unique_metadata_tuples = set(tuple(d.items()) for d in all_results_metadata)
    unique_metadata = [dict(t) for t in unique_metadata_tuples]
    
    full_context_articles = chunks_to_article(unique_metadata)
    
    # 5. Construir el prompt
    prompt = build_prompt(standalone_query, full_context_articles)
    
    # 6. Generar la respuesta
    answer = get_llm_response(prompt)
    
    if not answer:
        raise HTTPException(status_code=500, detail="Failed to generate a response from the LLM.")
        
    # 7. Devolver la respuesta en el formato correcto
    return ChatResponse(answer=answer, sources=unique_metadata)