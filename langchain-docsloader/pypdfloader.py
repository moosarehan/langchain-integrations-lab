from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("artificial_intelligence.pdf")
docs = loader.load()

print("Number of documents (pages):", len(docs))

print(docs[0].page_content)
print(docs[0].metadata)