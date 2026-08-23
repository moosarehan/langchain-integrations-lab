from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="fact1 about blackhole"),
    ResponseSchema(name="fact2", description="fact2 about blackhole"),
    ResponseSchema(name="fact3", description="fact3 about blackhole"),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template="give some facts about {topic} \n{format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. Chain & Invoke
chain = template1 | model | parser
response = chain.invoke({'topic':'blackhole'})

print(response)