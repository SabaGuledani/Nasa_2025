"""functions for llms"""
from __future__ import annotations
from google.genai import types, Client
from datetime import datetime

def run_llm(client:Client, system_instruction:str, messages:list[Content], llm_model_name:str,grounding:bool=True):
    """runs llm call, returns response object from gemini"""
    if grounding:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        # Use system instruction (cleaner than faking a system prompt)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[grounding_tool]
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    response = client.models.generate_content(model=llm_model_name,
                                              contents=messages,
                                              config=config)
    
    return response

def handle_router(client:Client, messages:list[Content], llm_model_name:str,routing_prompt:str) -> types.GenerateContentResponse:
    """get route to next destination for"""
    system_instructions = routing_prompt
    response = run_llm(client,
                       system_instructions,
                       messages,
                       llm_model_name,
                       False)
    
    return response


def handle_answer(client:Client, messages:list[Content], llm_model_name:str) -> types.GenerateContentResponse:
    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    system_instructions = "prompt"
    response = run_llm(client,
                       system_instruction=system_instructions,
                       messages=messages,
                       llm_model_name=llm_model_name)
    return response