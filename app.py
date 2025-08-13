# app.py  (verbose logging to show prompts/replies in console)
import os
import sys
import traceback
import logging
from flask import Flask, render_template, request, session
from dotenv import load_dotenv

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

# ---------- Logging (reliable on Windows PowerShell) ----------
# Force logging to stdout and flush immediately
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------

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

# ---------- Helpers ----------
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

def _get_msg_from_request() -> str:
    """Accept form or JSON payloads; return trimmed string."""
    msg = (request.form.get("msg") or "").strip()
    if not msg:
        data = request.get_json(silent=True) or {}
        msg = (data.get("msg") or "").strip()
    return msg

def _log_turn(user_msg: str, bot_msg: str | None = None):
    logger.info("USER: %s", user_msg)
    if bot_msg is not None:
        logger.info("BOT : %s", bot_msg)
# -----------------------------

@app.before_request
def log_incoming():
    # See every request path/method immediately in the console
    logger.info(">>> %s %s", request.method, request.path)

@app.route("/")
def index():
    return render_template("chat.html")

# ── original route (no memory)
@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = _get_msg_from_request()
        if not msg:
            logger.warning("Empty input received on /get")
            return "Empty input received."

        _log_turn(user_msg=msg)

        response = rag_chain.invoke({"input": msg})
        answer = (response.get("answer") or "").strip()
        if not answer:
            answer = "Sorry, no answer was found."

        _log_turn(user_msg=msg, bot_msg=answer)
        return answer
    except Exception:
        logger.exception("Unhandled error in /get")
        traceback.print_exc()
        return "An internal error occurred. Please try again later."

# ── memory route (uses prior turns)
@app.route("/memory-chat", methods=["POST"])
def memory_chat():
    try:
        msg = _get_msg_from_request()
        if not msg:
            logger.warning("Empty input received on /memory-chat")
            return "Empty input received."

        # save user turn
        hist = _get_history()
        hist.append({"role": "user", "content": msg})
        session.modified = True

        _log_turn(user_msg=msg)

        # call RAG with history injected
        response = memory_rag_chain.invoke({
            "input": msg,
            "history": _history_text(),
        })
        answer = (response.get("answer") or "").strip()
        if not answer:
            answer = "Sorry, no answer was found."

        # save assistant turn
        hist.append({"role": "assistant", "content": answer})
        session.modified = True

        _log_turn(user_msg=msg, bot_msg=answer)
        return answer
    except Exception:
        logger.exception("Unhandled error in /memory-chat")
        traceback.print_exc()
        return "An internal error occurred. Please try again later."

# optional: clear history
@app.route("/reset-history", methods=["POST"])
def reset_history():
    session.pop("history", None)
    logger.info("Chat history cleared for this session.")
    return "History cleared."

if __name__ == "__main__":
    # Run unbuffered; logging already flushes immediately
    app.run(host="0.0.0.0", port=8080, debug=True)
