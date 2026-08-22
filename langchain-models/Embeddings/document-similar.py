import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
   
)
docs = [
    "Faisalabad is known as the Manchester of Pakistan.",
    "breaking bad is one of the best televison shows in history",
    "the best cricket player is virat kohli",
    "erling haaaland is a man city player and a norwegian footballer",
    "AI applications are getting famous nowadays due to agentic ai /generative ai and it has impacted many areas such as customer support software development and education"


]
query='tell me about erling halaand'

result = embeddings.embed_documents(docs)
queryvect=embeddings.embed_query(query)
scores=(cosine_similarity([queryvect],result))[0]
print(scores)
