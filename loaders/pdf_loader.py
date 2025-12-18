from langchain_community.document_loaders import PyPDFium2Loader

def load_pdf(path):
    loader = PyPDFium2Loader(path)
    docs = loader.load()
    print(" Pages extracted:", len(docs))
    print(" Sample text:", docs[0].page_content[:500], "...\n")
    return docs
