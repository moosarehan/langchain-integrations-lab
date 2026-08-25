from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
  template='write a tweet about {topic}',
  input_variables=['topic']

)
prompt2 = PromptTemplate(
  template='write a linkedin post  about {topic}',
  input_variables=['topic']

)
parser = StrOutputParser()
chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)}
)


topic_input = {"topic": "AI"}

res1 = chain.invoke(topic_input)
print(res1)