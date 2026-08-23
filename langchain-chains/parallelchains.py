from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1=ChatGoogleGenerativeAI(
     model="gemini-3.6-flash"
)



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
)

# Wrap with ChatHuggingFace
model2 = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='generate brief notes on the following {text}',
    input_variables=['text']

)

prompt2=PromptTemplate(
    template='generate 5 question answer quiz on the following {text}',
    input_variables=['text']

)

prompt3=PromptTemplate(
    template='merge the  notes {notes} and the quiz question answer {quiz} into a single document ',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()
parallelchain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
}
)
merge= prompt3 | model1 | parser
chain=parallelchain | merge
text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.
Still effective in cases where number of dimensions is greater than the number of samples.
Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

"""
response=chain.invoke({'text':text})
print(response)