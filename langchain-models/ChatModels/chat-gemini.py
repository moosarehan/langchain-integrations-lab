import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# Load variables from .env
load_dotenv()

# Initialize standard LLM model
model = ChatGoogleGenerativeAI(
  model="gemini-3.6-flash"
)

# Run text completion
prompt = "Explain solid principles each prinicple in 2 lines"
response = model.invoke(prompt)

print(response.content)