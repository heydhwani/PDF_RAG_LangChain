from langchain_community.document_loaders import PyPDFLoader


pdf_path = "docs/rag_paper.pdf"


loader = PyPDFLoader(pdf_path)


documents = loader.load()


print("Total pages:", len(documents))


print("\nFirst page content:\n")
print(documents[0].page_content)