# store_index.py

from dotenv import load_dotenv
import os

# Import only what you need to avoid circular issues
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings 

# Load .env variables
load_dotenv()

# existing Pinecone index
index_name = "lawbot"

# Load embeddings
embedding_model = download_hugging_face_embeddings()

# Load the existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)
