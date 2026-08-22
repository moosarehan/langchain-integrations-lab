from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3.8-2.4T-A95B',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content='you are a professional doctor'),
    HumanMessage(content='tell me about lung cancer')
]
response=model.invoke(messages)
messages.append(AIMessage(content=response.content))
print(messages)