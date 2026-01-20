# # 🤖 RAG API - Retrieval Augmented Generation with LangChain & GPT-4o

This project is a **Retrieval-Augmented Generation (RAG)** API built using **LangChain**, **FastAPI**, and **OpenAI GPT-4o**.
It allows users to ask questions and get answers based on content retrieved from web documents.

---

## What this project does

* Loads documents from a web URL
* Splits text into smaller chunks
* Converts text into vector embeddings
* Retrieves relevant content using similarity search
* Generates answers using GPT-4o

---

## Tech Stack

* Python
* FastAPI
* LangChain
* OpenAI GPT-4o
* FAISS (Vector Database)
* HuggingFace Sentence Transformers
* BeautifulSoup
* Uvicorn

---

## How RAG Works (Simple)

1. Documents are loaded from a URL
2. Text is split into chunks
3. Chunks are converted into embeddings
4. FAISS finds relevant chunks for a question
5. GPT-4o generates the final answer

---

## API Endpoints

### GET `/`

Returns basic API information.

### GET `/health`

Checks if the API and RAG pipeline are running.

### POST `/index`

Indexes documents from a web URL.

**Request**

```json
{
  "url": "https://python.langchain.com/docs/get_started"
}
```

### POST `/ask`

Asks a question based on indexed documents.

**Request**

```json
{
  "question": "What is LangChain?"
}
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 3. Run the application

```bash
python main.py
```

### 4. Open API docs

* Swagger UI: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## Project Structure

```
Langchain_RAG_using_GPT4o/
├── main.py
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
├── README.md
└── .gitignore
```

---

## Use Cases

* Document Question Answering
* Knowledge base assistants
* Learning and experimentation with RAG
* Technical documentation Q&A

---

## Notes

* `.env` and virtual environments are not committed
* FAISS runs in memory
* Designed for learning and small to medium datasets

---

**Author:** Thrinadh Prasadapu
**Last Updated:** January 2026
