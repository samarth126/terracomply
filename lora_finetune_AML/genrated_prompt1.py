def prompt_template(data: str, num_records: int = 5):
    """
    Generates the instruction prompt for the Gemini model to create structured Q&A pairs.
    """
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model.

    **Goal:** Generate exactly {num_records} distinct, high-quality Q&A pairs based on the text provided below.

    **Instructions:**
    1. For each entry, generate one well-structured question reflecting different aspects of the information in the chunk.
    2. Ensure a mix of longer (up to 3-4 sentences) and shorter (1-2 sentences) questions.
    3. The answer must be concise yet informative, capturing key insights strictly from the text.
    4. The output must adhere strictly to the JSON schema provided in the API call configuration.

    Example Q&A structure (do not include this example in the final JSON output):
        "question": "What is the primary purpose of this dataset?",
        "answer": "This dataset serves as training data for fine-tuning a language model."
    
    Focus on creating clear, relevant, and varied questions that encourage the model to learn from diverse perspectives. 
    Avoid any sensitive or biased content, ensuring answers are accurate and neutral.

    --- DATA CHUNK ---
    {data}
    """


if __name__ == "__main__":
    # Example usage for testing the prompt structure
    print(prompt_template("This is a test data snippet about AI.", 3))
