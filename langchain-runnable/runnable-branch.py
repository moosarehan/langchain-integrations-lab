from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
  template='write a detailed report on  {topic}',
  input_variables=['topic']

)


prompt2 = PromptTemplate(
  template='summarize the report on   {topic}',
  input_variables=['topic']

)


parser = StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
chain2=RunnableBranch(
    (lambda x:len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
     
)
final=chain | chain2


topic_input = {"topic": "AI"}

res1 = final.invoke(topic_input)
print(res1)