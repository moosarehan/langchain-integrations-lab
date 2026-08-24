from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda


load_dotenv()

# 1. Initialize LLM & Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.1  # Lower temperature for deterministic schema adherence
)
model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=Feedback)

parser1=StrOutputParser()

template1 = PromptTemplate(
    template="Analyze the following feedback and determine if it is positive or negative.\n{format_instructions}\n\nFeedback: {feedback_text}",
    input_variables=["feedback_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain = template1 | model | parser
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "sentiment could not be found")
)

chain_classify=chain | branch_chain
response=chain_classify.invoke({'feedback_text':"s24 is a terrible smartphone"})
print(response)