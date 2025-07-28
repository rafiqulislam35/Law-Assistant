from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import traceback

# LangChain & Vector DB
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Local Modules
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# ────────────────────────────────────────────── #
# Load Environment Variables
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# ────────────────────────────────────────────── #
# Embeddings & Pinecone Vector Store Setup
embedding_model = download_hugging_face_embeddings()

index_name = "lawbot"  # Ensure this matches your Pinecone index name
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ────────────────────────────────────────────── #
# LLM, Prompt, and RAG Chain using Ollama
llm = Ollama(model="llama3", temperature=0.4)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ────────────────────────────────────────────── #
# Flask App Setup
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    print("🟢 Flask route triggered")

    try:
        msg = request.form.get("msg", "")
        if not msg.strip():
            return "⚠️ Empty input received."

        print(f"👤 User Input: {msg}")

        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer")

        if answer:
            print("✅ Answer:", answer)
            return str(answer)
        else:
            print("⚠️ No 'answer' key in response.")
            return "Sorry, no answer was found."

    except Exception as e:
        print("❌ Error during /get route:", e)
        traceback.print_exc()
        return "An internal error occurred. Please try again later."

# ────────────────────────────────────────────── #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
