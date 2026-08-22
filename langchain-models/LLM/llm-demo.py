import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

# Load variables from .env
load_dotenv()

# Initialize standard LLM model
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Run text completion
prompt = "Explain linear regression in 2 sentences"
response = llm.invoke(prompt)

print(response)