from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://helloclue.com/articles/sex/a-beginners-guide-to-sex")
docs = loader.load()

print(docs[0].page_content)