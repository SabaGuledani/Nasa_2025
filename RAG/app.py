from __future__ import annotations
from dotenv import load_dotenv
import streamlit as st
from llm import handle_answer, handle_router, run_llm
from google.genai import Client
from sentence_transformers import SentenceTransformer
import torch
from utils import get_history, clean_response
from prompts import ROUTING_SYSTEM_PROMPT, REFORMULATION_SYS_PROMPT, get_answer_prompt
import os
from pymilvus import connections, Collection, utility

load_dotenv()

# Set up page title and icon
st.set_page_config(page_title="Gemini Travel Assistant", page_icon="✈️")

LLM_MODEL_NAME = "gemini-2.5-flash"
ROUTER_LLM_MODEL_NAME = "gemini-2.0-flash-lite"


# Load environment variables
load_dotenv()

# Embedding model setup
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# Milvus connection args
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# Global connection
_connection = None

def connect_to_milvus():
    """Connect to Milvus server (singleton pattern)."""
    global _connection
    if _connection is None:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        _connection = True

def fix_collection_index(collection_name: str):
    """Check and fix the index for a collection."""
    connect_to_milvus()
    
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist")
        return
    
    collection = Collection(collection_name)
    
    # Check existing indexes
    print(f"\n=== Checking collection: {collection_name} ===")
    indexes = collection.indexes
    print(f"Current indexes: {indexes}")
    
    # Check the metric type of existing index
    if indexes:
        for idx in indexes:
            print(f"Index details: {idx.params}")
            metric_type = idx.params.get('metric_type', 'Unknown')
            print(f"Current metric type: {metric_type}")
            
            if metric_type == 'COSINE':
                print("Index already uses COSINE metric. No changes needed.")
                collection.load()
                return
    
    # Release collection before dropping index
    try:
        collection.release()
        print("Released collection")
    except Exception as e:
        print(f"Collection release note: {e}")
    
    # Drop existing index on embedding field
    try:
        collection.drop_index()
        print("Dropped existing index")
    except Exception as e:
        print(f"No index to drop or error: {e}")
    
    # Create new index with COSINE metric
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print(f"Created new index with COSINE metric type")
    
    # Verify the index
    indexes = collection.indexes
    print(f"New indexes: {indexes}")
    if indexes:
        for idx in indexes:
            print(f"New index metric type: {idx.params.get('metric_type', 'Unknown')}")
    
    # Load collection
    collection.load()
    print(f"Collection loaded successfully\n")

def search_collection(collection_name: str, query: str, output_fields: list, k: int = 5):
    """
    Generic search function for any Milvus collection.
    
    Args:
        collection_name: Name of the collection to search
        query: Search query text
        output_fields: List of field names to retrieve
        k: Number of results to return
    
    Returns:
        Raw search results from Milvus
    """
    connect_to_milvus()
    
    # Get collection
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = Collection(collection_name)
    collection.load()
    
    # Generate query embedding
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    # Search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    # Perform search
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=output_fields
    )
    
    return results[0]

def search_publications(query: str, k: int = 5):
    """Search the publications collection."""
    output_fields = ["PMC_code", "name", "authors", "date", "doi", "content"]
    results = search_collection("publications", query, output_fields, k)
    
    formatted_results = []
    for hit in results:
        formatted_results.append({
            "PMC_code": hit.entity.get("PMC_code"),
            "name": hit.entity.get("name"),
            "authors": hit.entity.get("authors"),
            "date": hit.entity.get("date"),
            "doi": hit.entity.get("doi"),
            "text": hit.entity.get("content"),
            "score": hit.score
        })
    
    return formatted_results

def search_osdr(query: str, k: int = 5):
    """Search the osdr collection."""
    output_fields = ["study_id", "name", "organisms", "authors", "doi", "link", "type", "protocole_name", "text"]
    results = search_collection("osdr", query, output_fields, k)
    
    formatted_results = []
    for hit in results:
        formatted_results.append({
            "study_id": hit.entity.get("study_id"),
            "name": hit.entity.get("name"),
            "organisms": hit.entity.get("organisms"),
            "authors": hit.entity.get("authors"),
            "doi": hit.entity.get("doi"),
            "link": hit.entity.get("link"),
            "type": hit.entity.get("type"),
            "protocol_name": hit.entity.get("protocole_name"),
            "text": hit.entity.get("text"),
            "score": hit.score
        })
    
    return formatted_results


client = Client()


# Set up Streamlit state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "itinerary" not in st.session_state:
    st.session_state.itinerary = {}

st.title("nasa")
query = st.chat_input("enter your question here")

def get_route(chat_history:list[dict]):
    """get json formatted route if rag is needed or not and which collections should be searched"""
    messages = get_history(chat_history)
    router_label = handle_router(client=client,
                              llm_model_name=ROUTER_LLM_MODEL_NAME,
                              messages=messages[-4:],
                              routing_prompt=ROUTING_SYSTEM_PROMPT)

    return clean_response(router_label.text)

def reformulate_prompt(prompt:str, chat_history:list[dict]):
    """reformulate user prompt to make it more searchable in rag, extract"""
    messages = get_history(chat_history)
    response = run_llm(client=client,
                        system_instruction=REFORMULATION_SYS_PROMPT,
                        messages=messages[-4:],
                        llm_model_name=ROUTER_LLM_MODEL_NAME,
                        grounding=False)

    return clean_response(response.text)

def run_search(reformulated_queries:dict):
    pubs_query = reformulated_queries.get("NASA_Space_Biology")
    osdr_query = reformulated_queries.get("Experiment_Collection")
    docs = []

    if pubs_query:
        publications = search_publications(query=pubs_query,k=4)
        for publication in publications:
            docs.append(publication)
    if osdr_query:
        osdrs = search_osdr(query=osdr_query,k=4) 
        for study in osdrs:
            docs.append(study)
    return docs

def get_llm_answer(chat_history):
    messages = get_history(chat_history)
    



if query:
    
    # Add user message to Streamlit chat
    st.session_state.chat_history.append({"role": "user", "content": query})

    chat_history = st.session_state.chat_history
    
    response_text = 
    
     # Add assistant message to Streamlit chat
    st.session_state.chat_history.append({"role": "model",
                                       "content": response_text})

for msg in st.session_state.chat_history:
    role = msg['role']
    if "model" in role:
        role = "asssistant"
    st.chat_message(role).write(msg['content'])