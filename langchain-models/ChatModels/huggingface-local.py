from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os


os.environ['HF_HOME']='E:/huggingface_cache'

llm=HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen3.8-2.4T-A95B',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.8
    )
)

model=ChatHuggingFace(llm=llm)
prompt='explain the beauty of Faislabad'
result=model.invoke(prompt)
print(result.content)
