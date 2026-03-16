from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



pdf_path = "docs/rag_paper.pdf"


loader = PyPDFLoader(pdf_path)


documents = loader.load()

print("Total pages:", len(documents))

print("\nFirst page content:\n")
print(documents[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0].page_content)