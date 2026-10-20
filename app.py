import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

DOCS_DIR = "docs"


def load_documents():
    docs = []
    for name in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, name)
        if name.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif name.lower().endswith((".txt", ".md")):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def answer(question, index, llm):
    results = index.similarity_search(question, k=4)
    context = "\n\n".join(r.page_content for r in results)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return llm.invoke(prompt).content, results


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        raise SystemExit("Put .pdf, .txt or .md files in the docs folder first.")
    index = build_index(docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print(f"Loaded {len(docs)} document pages. Ask a question (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        text, sources = answer(q, index, llm)
        print(f"\nAssistant: {text}")
        print("Sources:", {s.metadata.get("source") for s in sources})