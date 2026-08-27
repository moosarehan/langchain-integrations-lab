import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text = """
Artificial Intelligence refers to machines simulating human intelligence.
It includes tasks like learning, reasoning, and problem-solving.

Cricket is a bat-and-ball game played between two teams of eleven players.
It is especially popular in countries like India, Pakistan, and Australia.

Machine learning is a subset of AI that allows systems to learn from data
without being explicitly programmed for every scenario.
"""

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

docs = splitter.create_documents([text])
print(docs)