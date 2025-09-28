#!/usr/bin/env python3
"""
Simple interface for interacting with the fraud_detect_llama model on Ollama
"""

import requests
import json
from typing import Optional

def query_fraud_model(prompt: str, model: str = "fraud_detect_llama:latest") -> Optional[str]:
    """
    Send a prompt to the Ollama model and return the response.
    
    Args:
        prompt (str): The question or prompt to send to the model
        model (str): The model name (default: fraud_detect_llama:latest)
    
    Returns:
        str: The model's response, or None if there was an error
    """
    
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Get complete response at once
    }
    
    try:
        # Send request to Ollama
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def interactive_mode():
    """
    Interactive mode for testing the model
    """
    print("Fraud Detection Model Interface")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\nEnter your question: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                print("Please enter a valid question.")
                continue
            
            print("\nThinking...")
            response = query_fraud_model(prompt)
            
            if response:
                print(f"\nResponse:\n{response}")
            else:
                print("Sorry, I couldn't get a response from the model.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    # Test the function with a sample prompt
    sample_prompt = """A compliance officer at a real estate brokerage is unsure whether a particular transaction falls under anti-money laundering (AML) reporting obligations. The deal involves a buyer purchasing commercial property through a corporation registered abroad, with payment structured through multiple wire transfers from different accounts. According to AML regulations, how should the officer determine whether this qualifies as a "real estate transaction" requiring a suspicious activity report (SAR), and what key factors should they consider in making that determination?"""
    
    print("Testing with sample prompt...")
    result = query_fraud_model(sample_prompt)
    
    if result:
        print("Sample Response:")
        print(result)
        print("\n" + "="*50 + "\n")
    
    # Start interactive mode
    interactive_mode()