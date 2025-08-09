import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# Load CSV documents from a folder
def load_csv_documents(folder_path):
    csv_docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".csv"):
            loader = CSVLoader(file_path=file_path)
            csv_docs.extend(loader.load())
    return csv_docs


# Load PDF documents from a folder
def load_pdf_documents(folder_path):
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# Combine all documents from a folder (CSV first, then PDF)
def load_all_documents(folder_path):
    csv_docs = load_csv_documents(folder_path)
    pdf_docs = load_pdf_documents(folder_path)
    combined = csv_docs + pdf_docs
    print(f"Loaded {len(csv_docs)} CSV chunks and {len(pdf_docs)} PDF chunks")
    return combined


# Split documents into text chunks
def split_into_chunks(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Total final chunks: {len(chunks)}")
    return chunks


# Load HuggingFace embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Load existing Pinecone index
def load_existing_index(index_name, embedding_model):
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )
    return docsearch
