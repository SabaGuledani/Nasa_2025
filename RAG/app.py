from __future__ import annotations
from dotenv import load_dotenv
import streamlit as st
from llm import handle_answer, handle_router
from google.genai import Client
from llm import handle_router

load_dotenv('./llm_rag/.env')

# Set up page title and icon
st.set_page_config(page_title="Gemini Travel Assistant", page_icon="✈️")

LLM_MODEL_NAME = "gemini-2.5-flash"
ROUTER_LLM_MODEL_NAME = "gemini-2.0-flash-lite"
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# embedding_model = embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

client = Client()


# Set up Streamlit state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "itinerary" not in st.session_state:
    st.session_state.itinerary = {}

st.title("nasa")
query = st.chat_input("enter your question here")

if query:
    
    # Add user message to Streamlit chat
    st.session_state.chat_history.append({"role": "user", "content": query})

    chat_history = st.session_state.chat_history
    messages = get_history(chat_history)
    
    router_label = handle_router(client=client,
                              llm_model_name=ROUTER_LLM_MODEL_NAME,
                              messages=messages[-4:])
    
    
     # Add assistant message to Streamlit chat
    st.session_state.chat_history.append({"role": "model",
                                       "content": response_text})
    


for msg in st.session_state.chat_history:
    role = msg['role']
    if "model" in role:
        role = "asssistant"
    st.chat_message(role).write(msg['content'])