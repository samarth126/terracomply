import os
import json
import requests
import time
from typing import List 
from pydantic import BaseModel
from colorama import Fore, Style
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Import the user's prompt template function. 
from generated_prompt import prompt_template

# --- Pydantic Schema for Structured Output (Used for Python objects, not schema generation) ---

class Record(BaseModel):
    """Schema for a single generated Question-Answer pair."""
    question: str
    answer: str

class Response(BaseModel):
    """The root response schema containing a list of records."""
    generated: List[Record]

# --- Gemini API Configuration ---
# Using the stable Gemini Flash model.
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Set a safety limit for the input text size to avoid Bad Request errors due to context length.
# Adjust this lower if you still hit context errors, but 30k is generally safe for Flash.
MAX_CONTEXT_LENGTH = 30000 

def llm_call(data: str, num_records: int = 5) -> dict:
    """
    Calls the Gemini API to generate structured Q&A records from text data.
    
    Uses exponential backoff for retries and includes detailed error logging for 400 status codes.
    """
    api_key = os.environ.get('GOOGLE_API')
    if not api_key:
        print(Fore.RED + "Error: GOOGLE_API environment variable not found." + Style.RESET_ALL)
        return {"generated": []}

    # 1. Input Safety Check (Truncation)
    original_data_length = len(data)
    if original_data_length > MAX_CONTEXT_LENGTH:
        print(Fore.YELLOW + f"Warning: Context truncated from {original_data_length} to {MAX_CONTEXT_LENGTH} characters to prevent potential 400 errors." + Style.RESET_ALL)
        data = data[:MAX_CONTEXT_LENGTH]
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    # --- CRITICAL FIX: MANUALLY DEFINED SIMPLE JSON SCHEMA ---
    # The Gemini API requires a simplified schema and rejects Pydantic's full schema
    # which contains "$defs" and "$ref" keywords.
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "generated": {
                "type": "ARRAY",
                "description": "A list of generated question and answer pairs.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {
                            "type": "STRING",
                            "description": "A question derived from the document text."
                        },
                        "answer": {
                            "type": "STRING",
                            "description": "The answer to the question, based strictly on the document text."
                        }
                    },
                    "required": ["question", "answer"]
                }
            }
        },
        "required": ["generated"]
    }

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_template(data, num_records)
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            # Use a slightly higher temperature for creative generation (Q&A)
            "temperature": 0.5 
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(Fore.CYAN + f"-> Calling Gemini API (Attempt {attempt + 1})..." + Style.RESET_ALL)
            response = requests.post(f"{API_URL}?key={api_key}", headers=headers, json=payload)
            
            # Detailed Error Logging: Check status code before raising for status
            if response.status_code == 400:
                print(Fore.RED + "--------------------------------------------------------" + Style.RESET_ALL)
                print(Fore.RED + f"400 Bad Request Error. Detailed API response:" + Style.RESET_ALL)
                try:
                    # Print the response body if it's JSON
                    print(json.dumps(response.json(), indent=2))
                except json.JSONDecodeError:
                    # Print the raw text if it's not JSON
                    print(response.text)
                print(Fore.RED + "--------------------------------------------------------" + Style.RESET_ALL)

            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            
            # Extract the JSON string from the response
            json_text = result.get('candidates', [{}])[0]\
                              .get('content', {})\
                              .get('parts', [{}])[0]\
                              .get('text', '{}')
            
            # Print the raw generated JSON for debugging/visibility
            print(Fore.LIGHTBLUE_EX + "\n--- Generated JSON (first 200 chars) ---" + Style.RESET_ALL)
            print(json_text[:200] + ('...' if len(json_text) > 200 else ''))
            print(Fore.LIGHTBLUE_EX + "--------------------------------------\n" + Style.RESET_ALL)

            # Parse the JSON string into a Python dictionary
            return json.loads(json_text)

        except requests.exceptions.RequestException as e:
            print(Fore.RED + f"Request failed: {e}" + Style.RESET_ALL)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(Fore.YELLOW + f"Retrying in {wait_time} seconds..." + Style.RESET_ALL)
                time.sleep(wait_time)
            else:
                print(Fore.RED + "Max retries reached. Skipping chunk." + Style.RESET_ALL)
                return {"generated": []}
        except json.JSONDecodeError as e:
            print(Fore.RED + f"Failed to parse generated JSON: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + f"Raw response text was:\n{json_text}" + Style.RESET_ALL)
            return {"generated": []}

if __name__ == "__main__": 
    # Initialize the converter and chunker
    converter = DocumentConverter()
    # NOTE: Assumes 'fraud_detect.pdf' is available in the run directory
    doc = converter.convert("fraud_detect.pdf").document
    chunker = HybridChunker()
    
    # CRITICAL FIX: Convert the chunks iterator to a list so we can get its length
    chunks = list(chunker.chunk(dl_doc=doc))

    dataset = {}
    
    print(Fore.GREEN + f"Processing {len(chunks)} document chunks with {MODEL_NAME}..." + Style.RESET_ALL)
    
    for i, chunk in enumerate(chunks): 
            print(Fore.MAGENTA + f"\n--- Processing Chunk {i+1}/{len(chunks)} ---" + Style.RESET_ALL)
            print(Fore.YELLOW + f"Raw Text Preview:\n{chunk.text[:300]}…" + Style.RESET_ALL)
            
            # Contextualize the chunk for better Q&A generation
            enriched_text = chunker.contextualize(chunk=chunk)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text Preview:\n{enriched_text[:300]}…" + Style.RESET_ALL)

            # Call the LLM to generate structured Q&A data
            data = llm_call(
                enriched_text
            )
            
            # Store the generated data and the context used
            dataset[i] = {"generated": data.get("generated", []), "context": enriched_text}
            
            num_generated = len(dataset[i]["generated"])
            print(Fore.GREEN + f"Successfully generated {num_generated} records for Chunk {i+1}." + Style.RESET_ALL)

    # Save the final dataset
    output_filename = 'tm1data_gemini.json'
    with open(output_filename, 'w') as f: 
        json.dump(dataset, f, indent=2)
        
    print(Fore.GREEN + f"\n--- DONE ---" + Style.RESET_ALL)
    print(Fore.GREEN + f"Dataset saved to {output_filename} containing {len(dataset)} chunks." + Style.RESET_ALL)
