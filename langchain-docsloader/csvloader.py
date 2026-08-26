from langchain_community.document_loaders import CSVLoader

# Basic CSV Loader initialization
loader = CSVLoader(
    file_path="user.csv",
   
)

# Load data (each row becomes a separate Document)
documents = loader.load()

# Example output
print(f"Loaded {len(documents)} rows.")
print("\nFirst Document Content:\n")
print(documents[0].page_content)
print("\nMetadata:", documents[0].metadata)