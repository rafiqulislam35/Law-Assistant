# app.py  (memory-ready, keeps your old /get)
from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import os, traceback

# LangChain & Vector DB
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Local
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt, memory_prompt

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

embedding_model = download_hugging_face_embeddings()
index_name = "lawbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = Ollama(model="llama3", temperature=0.4)

base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
base_qa = create_stuff_documents_chain(llm, base_prompt)
rag_chain = create_retrieval_chain(retriever, base_qa)

# Memory chain (injects history explicitly)
memory_prompt_tpl = ChatPromptTemplate.from_messages([
    ("system", memory_prompt),
    ("system", "Chat history so far:\n{history}"),
    ("human", "{input}")
])
memory_qa = create_stuff_documents_chain(llm, memory_prompt_tpl)
memory_rag_chain = create_retrieval_chain(retriever, memory_qa)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")  # required for sessions

def _get_history() -> list:
    """
    Returns a list of {'role': 'user'|'assistant', 'content': str}.
    Stored per browser session.
    """
    if "history" not in session:
        session["history"] = []
    return session["history"]

def _history_text() -> str:
    hist = _get_history()
    return "\n".join(f"{h['role']}: {h['content']}" for h in hist)

@app.route("/")
def index():
    return render_template("chat.html")

# ── your original route (no memory)
@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "")
        if not msg.strip():
            return "Empty input received."
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer")
        return str(answer) if answer else "Sorry, no answer was found."
    except Exception as e:
        traceback.print_exc()
        return "An internal error occurred. Please try again later."

# ── memory route (uses prior turns)
@app.route("/memory-chat", methods=["POST"])
def memory_chat():
    try:
        msg = request.form.get("msg", "")
        if not msg.strip():
            return "Empty input received."

        # save user turn
        hist = _get_history()
        hist.append({"role": "user", "content": msg})
        session.modified = True

        # call RAG with history injected
        response = memory_rag_chain.invoke({
            "input": msg,
            "history": _history_text(),
        })
        answer = response.get("answer", "").strip() or "Sorry, no answer was found."

        # save assistant turn
        hist.append({"role": "assistant", "content": answer})
        session.modified = True

        return answer
    except Exception:
        traceback.print_exc()
        return "An internal error occurred. Please try again later."

# optional: clear history
@app.route("/reset-history", methods=["POST"])
def reset_history():
    session.pop("history", None)
    return "History cleared."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
