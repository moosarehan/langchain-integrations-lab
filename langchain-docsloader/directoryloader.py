from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all text files from a target directory (including subdirectories)
loader = DirectoryLoader(
    path="books",
    glob="*.pdf",          # Pattern matching (e.g., "*.md", "*.py", or "**/*.*" for all files)
    loader_cls=PyPDFLoader
)

# Load documents into memory
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")
print(documents[1].page_content)
print(documents[1].metadata)