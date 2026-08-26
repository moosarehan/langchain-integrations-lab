from langchain_community.document_loaders import TextLoader

# Load the clean.txt file located in the same directory
loader = TextLoader('clean.txt', encoding='utf-8')
docs = loader.load()

# Access document content and metadata
print("Total documents loaded:", len(docs))
print("\n--- Document Metadata ---")
print(docs[0].metadata)

print("\n--- Document Content ---")
print(docs[0].page_content)