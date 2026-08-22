from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3.8-2.4T-A95B',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
chathistory=[
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    prompt= input('You:')
    if prompt =='exit':
        break
    chathistory.append(HumanMessage(content=prompt))
    result= model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))
    print('AI:',result.content)



print(chathistory)
