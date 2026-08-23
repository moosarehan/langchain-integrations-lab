from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
)

# Wrap with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template='write detailed report on {topic}',
    input_variables=['topic']

)


template2=PromptTemplate(
    template='write a 5 pionter  summary on {text}',
    input_variables=['text']

)


parser=StrOutputParser()
chain=template1 | model | parser | template2 | model | parser

response=chain.invoke({'topic':'rise of ai in tech industry'})
print(response)