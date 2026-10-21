# AI Knowledge Assistant<img width="1280" height="640" alt="banner" src="https://github.com/user-attachments/assets/e6840232-740c-4901-b257-8716439cf506" />


A question-answering assistant that answers questions over your own documents using retrieval-augmented generation (RAG). Answers are grounded in your files and cite the source document they came from.

## How it works

1. Loads PDF, TXT, and Markdown files from the `docs/` folder
2. Splits them into overlapping chunks
3. Embeds the chunks with OpenAI embeddings and stores them in a FAISS vector index
4. Retrieves the most relevant chunks for each question
5. Passes them to an LLM that answers using only that context, or says it doesn't know

## Tech stack

Python, LangChain, FAISS, OpenAI API (embeddings and chat)

## Setup

```bash
git clone https://github.com/psrisaikiran24/AI-Knowledge-Assistant.git
cd AI-Knowledge-Assistant
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf python-dotenv
```

Create a `.env` file in the project folder:

```
OPENAI_API_KEY=your_key_here
```

## Usage

Put your PDF, TXT, or MD files in a `docs/` folder, then run:

```bash
python app.py
```

Example:

```
You: What programming languages does this person know?
Assistant: Python, JavaScript, TypeScript, SQL, Java, Bash/Shell.
Sources: {'docs\\resume.pdf'}
```

## Roadmap

- Streamlit web interface
- Anthropic Claude as an alternative model
- Persistent vector index
- Retrieval quality evaluation

## License

MIT
