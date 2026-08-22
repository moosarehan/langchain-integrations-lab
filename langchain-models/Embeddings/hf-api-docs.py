import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
   
)
docs = [
    "Faisalabad is known as the Manchester of Pakistan.",
    "It has a rich history in the textile industry.",
    "The city is famous for the Clock Tower (Ghanta Ghar)."
]
result = embeddings.embed_documents(docs)

print(str(result))
print("Embedding length:", len(result))