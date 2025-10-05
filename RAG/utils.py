from __future__ import annotations
import json
import re
import ast
from google.genai import types
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def save_json(data, filename):

    """
    Saves a Python object as a JSON file.

    Args:
        data (dict or list): The Python object to save.
        filename (str): The path to the file (e.g., 'data.json').
    """
    if not filename.endswith('.json'):
        filename += '.json'

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  # pretty print
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def load_json(filename: str) -> dict:
    """
    Load JSON data from a file and return it as a Python dictionary.
    
    :param filename: Path to the JSON file.
    :return: Dictionary with the loaded JSON data.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        return {}


def find_elem(selector:str, wd:webdriver, selector_method:str="xpath"):
    
    try:
        if selector_method == "xpath":
            # ✅ Wait until element is present AND visible (up to 10 seconds)
            element = WebDriverWait(wd, 10).until(
                EC.visibility_of_element_located((By.XPATH, selector))
            )
        elif selector_method == "css":
             # ✅ Wait until element is present AND visible (up to 10 seconds)
            element = WebDriverWait(wd, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
            )

        # ✅ Scroll into view (so it is 100% seen on screen)
        wd.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)

        return element

    except Exception as e:
        print(e)
        return None


def get_history(chat_history:list[dict]) -> list[Content]:
    '''parse chat history messages as Content type from chat_history list'''
    for msg in chat_history:
        print(f"content: {msg['content']}")
    messages = [
        types.Content(
            role=msg["role"],
            parts=[types.Part.from_text(text=msg["content"])]
        )
        for msg in chat_history
    ]

    return messages