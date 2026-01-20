PROJECT DOCUMENTATION
LangChain RAG API - Complete Documentation
📋 Table of Contents
Project Overview
Technology Stack
LLM Models
Chunking Technique
Embedding Technique
Vector Database
API Architecture
API Endpoints
How It Works
Setup & Installation
🎯 Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system that:
Loads documents from web URLs
Converts them into searchable vector embeddings
Retrieves relevant context based on user questions
Generates accurate answers using OpenAI's GPT-4o model
RAG (Retrieval-Augmented Generation) combines:
Information retrieval from a knowledge base
Language model generation capabilities
🛠️ Technology Stack
Core Frameworks & Libraries
Technology	Version/Package	Purpose
FastAPI	Latest	REST API framework
Uvicorn	uvicorn[standard]	ASGI server
LangChain	Latest	RAG orchestration
LangChain Community	Latest	Community integrations
LangChain OpenAI	Latest	OpenAI LLM integration
LangChain Text Splitters	Latest	Document chunking
FAISS	faiss-cpu	Vector database
HuggingFace Transformers	sentence-transformers	Embedding models
BeautifulSoup4	Latest	Web scraping
Python-dotenv	Latest	Environment variables
Pydantic	Latest	Data validation
Dependencies (requirements.txt)
fastapiuvicorn[standard]python-dotenvlangchainlangchain-communitylangchain-openailangchain-text-splitterslangchainhubfaiss-cpubeautifulsoup4pydanticsentence-transformers
🤖 LLM Models
Primary LLM: OpenAI GPT-4o
Model Details:
Provider: OpenAI
Model Name: gpt-4o
Type: Multimodal large language model
Configuration:
Temperature: 0 (deterministic responses)
Max Tokens: 500 (response length limit)
Use Case: Answer generation based on retrieved context
Why GPT-4o?
Strong reasoning
Good context understanding
Handles typos/variations
Reliable output quality
Alternative Models Available:
gpt-4o-mini (cheaper, faster)
gpt-4-turbo (previous generation)
gpt-3.5-turbo (budget option)
📄 Chunking Technique
Method: Recursive Character Text Splitter
Implementation:
text_splitter = RecursiveCharacterTextSplitter(    chunk_size=1000,      # Characters per chunk    chunk_overlap=200,    # Overlap between chunks    length_function=len,)
Configuration Details:
Chunk Size: 1000 characters
Balance between context and granularity
Fits embedding model limits
Chunk Overlap: 200 characters
Preserves context across boundaries
Prevents information loss at splits
Improves retrieval quality
How It Works:
Splits by separators (paragraphs, sentences, words)
Tries larger splits first, falls back to smaller ones
Maintains overlap for continuity
Respects character limits
Why This Method?
Preserves semantic meaning
Handles various document types
Configurable chunk sizes
Handles code and structured text well
🧮 Embedding Technique
Model: HuggingFace Sentence Transformers
Model Details:
Model Name: all-MiniLM-L6-v2
Provider: HuggingFace
Type: Sentence transformer
Dimensions: 384-dimensional vectors
Size: ~80MB (lightweight)
License: Apache 2.0
Implementation:
embeddings = HuggingFaceEmbeddings(    model_name="all-MiniLM-L6-v2")
Characteristics:
Fast inference
Good quality for many tasks
Local execution (no API calls)
Free and open-source
Works well for semantic search
How Embeddings Work:
Converts text → numeric vectors
Similar text → similar vectors
Enables semantic similarity search
Vectors stored in FAISS
Why This Model?
Good performance/speed balance
Supports various languages
Low resource usage
Well-suited for RAG
Alternative Embedding Options:
sentence-transformers/all-mpnet-base-v2 (higher quality)
OpenAIEmbeddings (OpenAI's embedding model)
text-embedding-ada-002 (via OpenAI API)
🗄️ Vector Database
Database: FAISS (Facebook AI Similarity Search)
Implementation:
vector = FAISS.from_documents(documents, embeddings)retriever = vector.as_retriever(    search_type="similarity",    search_kwargs={"k": 5}  # Retrieve top 5 chunks)
FAISS Details:
Type: In-memory vector database
Library: faiss-cpu (CPU version)
Search Method: Similarity search
Retrieval: Top K nearest neighbors (k=5)
Features:
Fast similarity search
Scalable indexing
Efficient for dense vectors
Local storage (no external database)
Configuration:
Search Type: similarity (cosine similarity)
K Value: 5 (retrieves top 5 most relevant chunks)
Storage: In-memory (loaded at startup)
How It Works:
Documents → embeddings → vectors
Vectors stored in FAISS index
Query → embedding → search
Returns top K similar vectors
Advantages:
Fast retrieval (milliseconds)
No external dependencies
Good for medium-sized datasets
Free and open-source
Limitations:
In-memory only (lost on restart)
Not persistent (can be saved/loaded)
Best for single-server deployments
🌐 API Architecture
Framework: FastAPI
Why FastAPI?
High performance
Automatic API documentation
Type validation with Pydantic
Async support
Modern Python features
Server Configuration:
app = FastAPI(    title="LangChain RAG API",    description="A RAG API using LangChain and OpenAI",    version="1.0.0")
CORS Middleware:
app.add_middleware(    CORSMiddleware,    allow_origins=["*"],  # Allows all origins    allow_credentials=True,    allow_methods=["*"],    allow_headers=["*"],)
CORS Setup:
Allows cross-origin requests
Useful for frontend integration
Configurable security
Server Details:
Host: 0.0.0.0 (all interfaces)
Port: 8001
Server: Uvicorn ASGI server
Auto-docs: Available at /docs (Swagger UI)
📡 API Endpoints
1. GET / - Root Endpoint
Purpose: Welcome message and API info
Response:
{  "message": "Welcome to LangChain RAG API",  "docs": "/docs",  "endpoints": {    "ask": "POST /ask - Ask a question",    "index": "POST /index - Index documents from a URL",    "health": "GET /health - Health check"  }}
2. GET /health - Health Check
Purpose: Check API status and RAG initialization
Response:
{  "status": "healthy",  "rag_chain_initialized": true/false}
Use Cases:
Health monitoring
Verify initialization status
Service health checks
3. POST /ask - Ask a Question
Purpose: Get answers based on indexed documents
Request Body:
{  "question": "What is LangChain?"}
Response:
{  "question": "What is LangChain?",  "answer": "LangChain is a framework for building..."}
Error Responses:
503 Service Unavailable: RAG chain not initialized
500 Internal Server Error: Processing error
How It Works:
Receives question
Retrieves relevant chunks (k=5)
Formats context
Generates answer via GPT-4o
Returns response
4. POST /index - Index Documents
Purpose: Load and index documents from a URL
Request Body:
{  "url": "https://python.langchain.com/docs/get_started"}
Response:
{  "message": "Documents indexed successfully",  "documents_indexed": 45}
How It Works:
Loads documents from URL
Splits into chunks
Generates embeddings
Stores in FAISS
Updates RAG chain
Default URL:
On startup: https://python.langchain.com/docs/get_started/introduction
🔄 How It Works (RAG Pipeline)
Initialization Phase:
Document Loading
WebBaseLoader fetches web content
Extracts text using BeautifulSoup4
Text Chunking
RecursiveCharacterTextSplitter splits documents
Creates chunks of 1000 chars with 200 char overlap
Embedding Generation
HuggingFaceEmbeddings converts chunks to vectors
384-dimensional embeddings per chunk
Vector Storage
FAISS indexes all embeddings
Enables fast similarity search
Retriever Setup
Configures similarity search
Returns top 5 most relevant chunks
LLM Initialization
GPT-4o model loaded
Prompt template configured
RAG Chain Creation
Links: Retrieval → Formatting → Prompting → LLM → Parsing
Query Phase:
User Question → API receives question
Question Embedding → Question converted to vector
Similarity Search → FAISS finds top 5 similar chunks
Context Formatting → Chunks formatted into context
Prompt Creation → Context + Question → Prompt
LLM Generation → GPT-4o generates answer
Response → Answer returned to user
LangChain Expression Language (LCEL) Chain:
rag_chain = (    {"context": retriever | format_docs, "question": RunnablePassthrough()}    | prompt    | llm    | StrOutputParser())
Flow:
Input: User question
Context retrieval: Retriever → format_docs
Prompt: Context + question → prompt template
LLM: Generate answer
Parse: Convert to string
Output: Final answer
🚀 Setup & Installation
Prerequisites:
Python 3.8+
OpenAI API key
Internet connection (for document loading)
Installation Steps:
Install Dependencies:
pip install -r requirements.txt
Configure Environment:
Create .env file:
OPENAI_API_KEY=your_openai_api_key_here
Run the Server:
python main.py
Access API Documentation:
Swagger UI: http://localhost:8001/docs
ReDoc: http://localhost:8001/redoc
Project Structure:
Langchain_RAG_using_GPT4o/├── main.py              # Main application code├── requirements.txt     # Python dependencies├── .env                 # Environment variables└── .gitignore          # Git ignore file
💡 Key Features
✅ Web-based document loading
✅ Intelligent text chunking
✅ Semantic similarity search
✅ GPT-4o powered answers
✅ RESTful API
✅ Auto-generated documentation
✅ CORS enabled
✅ Error handling
✅ Health monitoring
✅ Dynamic document indexing
🎓 Use Cases
Document Q&A
Internal knowledge bases
Documentation systems
FAQ bots
Research Assistance
Academic paper analysis
Technical documentation
Content summarization
Customer Support
Help desk automation
Product documentation
Troubleshooting guides
Education
Study assistants
Learning platforms
Educational content
📊 Performance Characteristics
Initialization: 10-30 seconds (depends on document size)
Query Response: 2-5 seconds (depends on LLM latency)
Embedding Generation: ~100ms per chunk
Vector Search: <10ms (FAISS retrieval)
Chunk Processing: Real-time indexing
🔐 Security Considerations
API keys stored in .env (not committed)
CORS configurable (currently allows all origins)
Input validation via Pydantic
Error handling prevents information leakage
📝 Conclusion
This RAG system uses:
GPT-4o for generation
Recursive Character Text Splitter for chunking
HuggingFace all-MiniLM-L6-v2 for embeddings
FAISS for vector storage
FastAPI for the REST API
The system retrieves relevant context and generates accurate answers based on indexed documents.
Document Version: 1.0
Last Updated: January 2026
Project: LangChain RAG using GPT-4o
Save this as PROJECT_DOCUMENTATION.md in your project folder.
