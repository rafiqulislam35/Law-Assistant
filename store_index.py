from src.helper import load_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Define index name (must match your existing Pinecone index)
index_name = "lawbot"  # <-- update if your actual index name is different

# Load embedding model (uses HuggingFaceEmbeddings)
embedding_model = load_huggingface_embeddings()

# Connect to the existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)
