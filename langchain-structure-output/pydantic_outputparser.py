from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)
class person(BaseModel):
    name:str
    age:int=Field(gt=18)

parser = PydanticOutputParser(pydantic_object=person)

template1 = PromptTemplate(
    template="give name and age of a fictional person  \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. Chain & Invoke
chain = template1 | model | parser
response = chain.invoke({})

print(response)