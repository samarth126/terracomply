import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# 1. Load the GOOGLE_API_KEY from the .env file
load_dotenv()

# The key is now loaded into os.environ, 
# but we'll still check it for clear feedback.
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: The GOOGLE_API_KEY was not found in the environment (or .env file).")
else:
    print("✅ GOOGLE_API_KEY successfully loaded from .env.")
    
    # 2. Attempt a simple API call
    try:
        # The client automatically picks up the GOOGLE_API_KEY from the environment
        client = genai.Client()

        print("Attempting a small API call to test the connection...")

        # Use a fast, low-cost model for a quick check
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Generate a single word that means "excellent".',
        )

        print("-" * 30)
        print("✅ SUCCESS: API key is valid and connection is working.")
        print(f"Model used: gemini-2.5-flash")
        print(f"API Response text: {response.text.strip()}")
        print("-" * 30)

    except APIError as e:
        print("\n❌ API ERROR: Connection failed. Check your API key validity and network connection.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")