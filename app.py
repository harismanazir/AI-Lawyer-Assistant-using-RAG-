import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import re

# ========== Setup ==========
load_dotenv()  # On Render, GROQ_API_KEY should be set as an env var in the dashboard

# Minimal sanity check for key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY not found. Set it in Render's dashboard (Environment → Add Environment Variable)."
    )

# Create necessary directories if missing
PDFS_DIR = "pdfs"
VSTORE_DIR = "vectorstore/db_faiss"
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VSTORE_DIR), exist_ok=True)

# Embeddings (free, no key needed). This downloads the model once on first run.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,model_kwargs={"device": "cpu"})

# LLM on Groq (free tier)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# Prompt
PROMPT_TMPL = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, say you don't know. Do NOT fabricate facts.
Answer strictly from the given context.

Question: {question}
Context:
{context}

Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)
parser = StrOutputParser()
chain = prompt | llm | parser

# ========== Helpers ==========

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()


def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def make_or_load_vectorstore(docs):
    # Build vectorstore from documents in memory; you can also cache with save/load if desired
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def get_context(docs):
    return "\n\n".join(d.page_content for d in docs)



# Cleaning utility to hide model reasoning when desired

def clean_response(response: str) -> str:
    """
    Removes <think>...</think> blocks from Groq/DeepSeek responses.
    Falls back to raw text if removal yields empty content.
    """
    if not response:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned if cleaned else response.strip()

# ========== UI ==========
st.set_page_config(page_title="AI Lawyer (Free) — Groq + HF", page_icon="⚖️")
st.title("⚖️ AI Lawyer (Free) — RAG on your PDFs")
st.caption("LLM: Groq DeepSeek R1 70B • Embeddings: all-MiniLM-L6-v2 • Vector DB: FAISS • Host: Render")

# Sidebar display options
a, b = st.columns([1, 3])
with st.sidebar:
    st.header("Display Options")
    show_trace = st.checkbox("Show reasoning trace (<think>…)", value=False)

uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"], accept_multiple_files=False)
user_query = st.text_area("Enter your question:", height=140, placeholder="Ask anything based on the uploaded PDF…")
ask = st.button("Ask AI Lawyer")

# Keep vectorstore and doc state across interactions
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if uploaded_file is not None:
    # Save to disk (ephemeral on Render, but fine for session)
    pdf_path = os.path.join(PDFS_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading and indexing your PDF…"):
        docs = load_pdf(pdf_path)
        chunks = split_docs(docs)
        st.session_state.vectorstore = make_or_load_vectorstore(chunks)
        st.session_state.doc_name = uploaded_file.name

    st.success(f"Indexed: {uploaded_file.name}")

if ask:
    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF first.")
    elif not user_query.strip():
        st.error("Please type a question.")
    else:
        with st.spinner("Retrieving relevant context and generating an answer…"):
            top_k_docs = st.session_state.vectorstore.similarity_search(user_query, k=5)
            context = get_context(top_k_docs)
            try:
                answer = chain.invoke({"question": user_query, "context": context})
            except Exception as e:
                st.exception(e)
                answer = None

        st.chat_message("user").write(user_query)
        if answer:
            rendered = answer if show_trace else clean_response(answer)
            st.chat_message("assistant").write(rendered)
        else:
            st.chat_message("assistant").write(
                "I couldn't generate an answer. Please verify your GROQ_API_KEY and try again."
            )

st.divider()
st.markdown(
    "**Tips**: Upload a focused legal PDF; ask specific questions. First load may take a bit while the embedding model downloads."
)
