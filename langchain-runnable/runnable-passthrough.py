from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
  template='write a joke about {topic}',
  input_variables=['topic']

)

prompt2 = PromptTemplate(
  template='write a 2 line explanation about {topic}',
  input_variables=['topic']

)
parser = StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
chain2=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':prompt2 | model | parser }
)
final=chain | chain2


topic_input = {"topic": "AI"}

res1 = final.invoke(topic_input)
print(res1)