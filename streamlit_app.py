import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚")
st.title("📚 AI Knowledge Assistant")
st.caption("Upload documents and ask questions. Answers come only from your files.")


@st.cache_resource(show_spinner="Indexing documents...")
def build_index(files_data):
    docs = []
    for name, data in files_data:
        suffix = os.path.splitext(name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        if suffix == ".pdf":
            loaded = PyPDFLoader(path).load()
        else:
            loaded = TextLoader(path, encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = name
        docs.extend(loaded)
        os.remove(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


uploads = st.file_uploader(
    "Upload PDF, TXT or MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploads:
    files_data = tuple((f.name, f.getvalue()) for f in uploads)
    index = build_index(files_data)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    question = st.text_input("Ask a question about your documents")
    if question:
        results = index.similarity_search(question, k=4)
        context = "\n\n".join(r.page_content for r in results)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt).content
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for name in sorted({r.metadata.get("source", "unknown") for r in results}):
            st.write(f"- {name}")
else:
    st.info("Upload at least one document to get started.")