import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Faisalabad is known for its textile industry and vibrant culture."
result = embeddings.embed_query(text)

print(result)
print("Embedding length:", len(result))