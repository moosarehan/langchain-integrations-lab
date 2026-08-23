from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

# 1. Initialize JsonOutputParser
parser = JsonOutputParser()

template1 = PromptTemplate(
    template="give name age city of fictional character \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. Chain & Invoke
chain = template1 | model | parser
response = chain.invoke({})

print(response)