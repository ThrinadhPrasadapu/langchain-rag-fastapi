from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

rag_chain = None
retriever = None


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str


class IndexRequest(BaseModel):
    url: str


class IndexResponse(BaseModel):
    message: str
    documents_indexed: int


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_rag_chain(url: str = "https://python.langchain.com/docs/get_started/introduction"):
    global rag_chain, retriever
    
    # Load documents from web
    loader = WebBaseLoader(url)
    data = loader.load()
    
    # Split documents into chunks with better configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=200,  # Overlap between chunks for better context
        length_function=len,
    )
    documents = text_splitter.split_documents(data)
    
    # Create vector store with FAISS using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector = FAISS.from_documents(documents, embeddings)
    
    # Configure retriever to get more relevant documents
    retriever = vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
      # Initialize LLM with OpenAI
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0,
        max_tokens=500  # Limit response length to prevent verbosity
    )
    
    # Concise prompt to avoid repetition
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the provided context to answer the question. Even if the question wording is slightly different (like "Lang chain" vs "LangChain"), try to answer based on the context. Be informative and clear.

Context: {context}

Question: {question}

Based on the context above, provide a helpful answer:"""
    )
    
    # Create RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return len(documents)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG chain...")
    try:
        num_docs = initialize_rag_chain()
        print(f"RAG chain initialized with {num_docs} documents")
    except Exception as e:
        print(f"Warning: Failed to initialize RAG chain on startup: {e}")
        print("You can initialize it later using the /index endpoint")
    yield
    print("Shutting down...")


app = FastAPI(
    title="LangChain RAG API",
    description="A RAG (Retrieval Augmented Generation) API using LangChain and OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to LangChain RAG API",
        "docs": "/docs",
        "endpoints": {
            "ask": "POST /ask - Ask a question",
            "index": "POST /index - Index documents from a URL",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_chain_initialized": rag_chain is not None
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized. Please index documents first using /index endpoint"
        )
    
    try:
        result = rag_chain.invoke(request.question)
        return QuestionResponse(
            question=request.question,
            answer=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    try:
        num_docs = initialize_rag_chain(request.url)
        return IndexResponse(
            message="Documents indexed successfully",
            documents_indexed=num_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)