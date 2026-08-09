import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config


def load_documents():
    loader = DirectoryLoader(
        config.DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {config.DATA_DIR}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
        collection_name="documind_collection",
    )
    print(f"Vector store built and persisted to {config.VECTORSTORE_DIR}")
    return vectordb


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No PDFs found in data/raw/. Add some and re-run.")
        exit()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete. You can now run: streamlit run app.py")