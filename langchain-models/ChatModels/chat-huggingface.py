from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3.8-2.4T-A95B',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
prompt='explain the beauty of Faislabad'
result=model.invoke(prompt)
print(result.content)
