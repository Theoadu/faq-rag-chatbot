# FAQ RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot for answering HR frequently asked questions using vector search and large language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Logging & Monitoring](#logging--monitoring)
- [Security](#security)
- [License](#license)

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that provides accurate, context-aware answers to HR policy questions. Instead of relying solely on the LLM's pre-trained knowledge, the system retrieves relevant document chunks from a vector database and uses them as context for generating responses.

### What is RAG?

RAG combines the power of:
1. **Retrieval**: Finding relevant information from a knowledge base using semantic search
2. **Augmentation**: Injecting that context into the LLM prompt
3. **Generation**: Having the LLM generate an answer based on the retrieved context

This approach ensures:
- ✅ **Factual accuracy** - Answers are grounded in actual documents
- ✅ **Reduced hallucinations** - LLM can't make up information
- ✅ **Source transparency** - System returns the chunks used
- ✅ **Easy updates** - Update knowledge base without retraining models

### Use Case: HR FAQ Automation

The chatbot answers employee questions about:
- Paid time off (PTO) policies
- Remote work guidelines
- Health insurance enrollment
- Expense reimbursement procedures
- Parental leave policies
- Professional development budgets
- Performance reviews
- And more...

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Indexing   │     │   Retrieval  │     │  Generation  │
│    Phase     │     │     Phase    │     │    Phase     │
└──────────────┘     └──────────────┘     └──────────────┘

1️⃣ INDEXING (Offline)
┌─────────────────┐
│  FAQ Document   │
│   (Text File)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────────┐
│  Text Splitter  │───>│  Document Chunks     │
│  (Recursive)    │    │  (300 chars each)    │
└─────────────────┘    └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Embedding Model     │
                       │  text-embedding-3    │
                       │  -small              │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   ChromaDB           │
                       │   Vector Store       │
                       │   (Persistent)       │
                       └──────────────────────┘


2️⃣ QUERY PROCESSING (Online)
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Moderation    │─── Unsafe? ──> ❌ Block & Return Error
│   (OpenAI API)  │
└────────┬────────┘
         │ Safe ✅
         ▼
┌─────────────────┐
│  Embed Question │
│  (Same Model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │
│  (ChromaDB)     │
│  Top K=2 chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────────┐
│  Context +      │───>│  LLM (gpt-4o-mini)   │
│  Question       │    │  Generate Answer     │
└─────────────────┘    └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   JSON Response      │
                       │   + Retrieved Chunks │
                       └──────────────────────┘
```

### Component Breakdown

#### 1. **Indexing Pipeline** (`src/build_index.py`)
- **Document Loader**: Reads FAQ text file
- **Text Splitter**: Uses `RecursiveCharacterTextSplitter` with:
  - Chunk size: 300 characters
  - Overlap: 50 characters
  - Separators: `\n\n`, `\n`, `. `, ` `
- **Embedding Generator**: OpenAI `text-embedding-3-small` model
- **Vector Store**: ChromaDB with persistent storage
- **Validation**: Ensures minimum 20 chunks generated

#### 2. **Query Pipeline** (`src/query.py`)
- **Input Validation**: Checks for empty questions
- **Content Moderation**: OpenAI Moderation API blocks unsafe inputs
- **Semantic Retrieval**: Finds top K=2 most relevant chunks
- **Answer Generation**: Uses `gpt-4o-mini` with:
  - Temperature: 0.0 (deterministic)
  - Max tokens: 250
  - System prompt enforcing context-only answers
- **Response Format**: Structured JSON with answer and source chunks

#### 3. **Evaluation System** (`src/evaluator.py`)
- **LLM-as-Judge**: Uses GPT-4o-mini to score answers (0-10)
- **Criteria**: Relevance, Accuracy, Completeness
- **Output**: Score + reasoning explanation
- **Format**: Enforced JSON output for consistency

#### 4. **Utilities** (`src/utils.py`)
- **Logging**: File + console logging with configurable levels
- **Moderation**: Safety check wrapper for OpenAI API
- **Configuration**: Environment-based setup

### Data Flow

**Indexing Flow:**
1. Load `faq_document.txt` (3,700+ characters)
2. Split into 20+ chunks using recursive splitter
3. Generate embeddings using OpenAI API
4. Store vectors in ChromaDB at `data/chroma/`
5. Save chunks as JSON for inspection

**Query Flow:**
1. User submits question via CLI
2. Moderation API checks for unsafe content
3. Question is embedded using same model
4. ChromaDB returns 2 most similar chunks
5. LLM generates answer using chunks as context
6. System returns JSON with answer + source chunks
7. All interactions logged to `logs/app.log`

### Design Patterns

- **RAG Pattern**: Retrieval-Augmented Generation for factual responses
- **Embedding Consistency**: Same model for indexing and retrieval
- **Fail-Safe Moderation**: Blocks unsafe content before processing
- **Structured Output**: JSON responses for easy integration
- **Separation of Concerns**: Distinct modules for build/query/evaluate

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.13+ | Primary development language |
| **LLM Model** | GPT-4o-mini | Latest | Answer generation |
| **Embedding Model** | text-embedding-3-small | Latest | Vector embeddings |
| **Vector Database** | ChromaDB | 1.3.4+ | Semantic search and storage |
| **Framework** | LangChain | 1.0.7+ | RAG orchestration |
| **Package Manager** | uv | Latest | Fast dependency management |

### Key Dependencies

#### LangChain Ecosystem
- **langchain** (1.0.7+): Core framework for LLM applications
- **langchain-openai** (1.0.3+): OpenAI integration (embeddings + LLM)
- **langchain-chroma** (1.0.0+): ChromaDB vector store integration
- **langchain-text-splitters** (1.0.0+): Document chunking utilities

#### OpenAI
- **openai** (2.8.0+): Official Python SDK for GPT models and embeddings

#### Vector Storage
- **chromadb** (1.3.4+): Embedded vector database with persistence

#### Utilities
- **python-dotenv** (1.2.1+): Environment variable management
- **numpy** (2.3.4+): Numerical operations for embeddings

### Infrastructure

- **Environment**: `.env` file for API keys and configuration
- **Version Control**: Git with `.gitignore` for secrets
- **Logging**: File-based logging to `logs/app.log`
- **Data Storage**: Local filesystem for ChromaDB and chunks
- **Testing**: pytest framework

## ✨ Features

### 1. **RAG-Powered Q&A**
- Semantic search across HR documentation
- Context-grounded answer generation
- Returns source chunks for transparency

### 2. **Content Moderation**
- OpenAI Moderation API integration
- Blocks harmful, hateful, or inappropriate queries
- Detailed category logging for flagged content

### 3. **Persistent Vector Store**
- ChromaDB with disk persistence
- Fast similarity search
- No need to rebuild index on restart

### 4. **Evaluation Framework**
- LLM-as-judge scoring system
- Automated quality assessment (0-10 scale)
- Evaluates relevance, accuracy, completeness

### 5. **Comprehensive Logging**
- File and console logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Tracks queries, retrievals, and errors

### 6. **JSON API Ready**
- Structured JSON output
- Easy integration with web services
- Error handling with consistent format

### 7. **Chunk Size Optimization**
- 300-character chunks with 50-char overlap
- Recursive splitting for semantic coherence
- Prevents information loss at boundaries

## 📁 Project Structure

```
faq-rag-chatbot/
│
├── src/                          # Source code
│   ├── build_index.py           # Indexing pipeline (offline)
│   ├── query.py                 # Query pipeline (online)
│   ├── evaluator.py             # Answer quality evaluation
│   └── utils.py                 # Shared utilities (logging, moderation)
│
├── data/                         # Data directory
│   ├── faq_document.txt         # HR FAQ source document
│   ├── chunks.json              # Exported text chunks (for inspection)
│   └── chroma/                  # ChromaDB persistent storage (gitignored)
│
├── logs/                         # Application logs (gitignored)
│   └── app.log                  # Runtime logs
│
├── tests/                        # Unit tests
│   ├── conftest.py              # pytest configuration
│   └── test_core.py             # Core functionality tests
│
├── .env                          # Environment variables (local, gitignored)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── .python-version               # Python version specification
├── main.py                       # Simple entry point
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Locked dependency versions
└── README.md                     # This file
```

### Key Files

- **`src/build_index.py`**: Builds the vector index from FAQ document
- **`src/query.py`**: Main query interface for answering questions
- **`src/evaluator.py`**: Evaluates answer quality using LLM
- **`src/utils.py`**: Logging setup and content moderation
- **`data/faq_document.txt`**: HR policy knowledge base (3,700+ chars)
- **`pyproject.toml`**: Dependency specifications

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**: Required for latest language features
- **uv**: Fast Python package manager ([installation guide](https://github.com/astral-sh/uv))
- **OpenAI API Key**: For embeddings and LLM ([get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd faq-rag-chatbot
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```
   This creates a virtual environment and installs all dependencies from `uv.lock`.

3. **Set up environment variables:**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-4o-mini
   LOG_LEVEL=INFO
   ```

4. **Build the vector index:**
   ```bash
   uv run python src/build_index.py
   ```
   
   This will:
   - Load `data/faq_document.txt`
   - Split into chunks
   - Generate embeddings
   - Store in ChromaDB at `data/chroma/`
   - Save chunks to `data/chunks.json`

5. **Verify installation:**
   ```bash
   uv run python src/query.py --question "What is the PTO policy?"
   ```

## 📖 Usage

### Command Line Interface

**Ask a question:**
```bash
uv run python src/query.py --question "How many days of PTO do employees get?"
```

**Response format:**
```json
{
  "user_question": "How many days of PTO do employees get?",
  "system_answer": "Full-time employees accrue 15 days of PTO per year.",
  "chunks_related": [
    "Paid Time Off (PTO) Policy\nFull-time employees accrue 15 days...",
    "...PTO begins accruing on the first day of employment..."
  ]
}
```

### Programmatic Usage

```python
from src.query import retrieve_relevant_chunks, generate_answer

# Retrieve relevant chunks
question = "What is the remote work policy?"
chunks = retrieve_relevant_chunks(question, k=2)

# Generate answer
answer = generate_answer(question, chunks)

print(f"Answer: {answer}")
print(f"Sources: {len(chunks)} chunks")
```

### Rebuilding the Index

If you update `data/faq_document.txt`, rebuild the index:

```bash
uv run python src/build_index.py
```

### Example Questions

- "How do I enroll in health insurance?"
- "What is the parental leave policy?"
- "How do I submit an expense report?"
- "What is the professional development budget?"
- "How often are performance reviews conducted?"

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_core.py::test_chunk_count

# Run with coverage
uv run pytest --cov=src
```

### Test Coverage

Current test suite validates:
- ✅ Document loading (>1000 characters)
- ✅ Chunk generation (≥20 chunks)
- ✅ Chunk content integrity ("PTO" in chunks)

### Manual Testing

```bash
# Test safe query
uv run python src/query.py --question "What is the PTO policy?"

# Test empty query
uv run python src/query.py --question ""

# Test moderation (should be blocked)
uv run python src/query.py --question "How to hack the system?"
```

## 📊 Evaluation

### Evaluate Answer Quality

```python
from src.evaluator import evaluate_answer
from src.query import retrieve_relevant_chunks, generate_answer

question = "What is the PTO policy?"
chunks = retrieve_relevant_chunks(question)
answer = generate_answer(question, chunks)

result = evaluate_answer(question, answer, chunks)
print(f"Score: {result['score']}/10")
print(f"Reason: {result['reason']}")
```

**Example output:**
```json
{
  "score": 9,
  "reason": "Answer accurately reflects chunk content, includes key details about 15 days PTO and 90-day waiting period. Minor: could mention non-rollover policy."
}
```

### Evaluation Criteria

1. **Relevance (0-10)**: Do chunks address the question?
2. **Accuracy (0-10)**: Does answer match chunk content?
3. **Completeness (0-10)**: Is all key info included?

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `LLM_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

### Adjusting Chunk Size

In `src/build_index.py`:

```python
chunks = chunk_text(
    text, 
    chunk_size=300,    # Increase for more context per chunk
    chunk_overlap=50   # Increase to reduce boundary loss
)
```

### Changing Retrieval Count

In `src/query.py`:

```python
chunks = retrieve_relevant_chunks(question, k=2)  # Change k for more/fewer chunks
```

### Modifying the Prompt

In `src/query.py`, edit the `generate_answer()` function:

```python
prompt = (
    "You are an HR support assistant. Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say: 'I don't know based on the provided documentation.'\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n"
    "Answer:"
)
```

## 📝 Logging & Monitoring

### Log Files

All activity is logged to `logs/app.log`:

```
2025-11-16 15:56:22 [INFO] RAGChatbot: Loaded document: 3742 characters
2025-11-16 15:56:22 [INFO] RAGChatbot: Created 23 chunks.
2025-11-16 15:56:24 [INFO] RAGChatbot: ChromaDB index saved to data/chroma
2025-11-16 16:02:15 [INFO] RAGChatbot: Processing query: What is the PTO policy?
2025-11-16 16:02:16 [DEBUG] RAGChatbot: Retrieved 2 chunks for query: What is the PTO...
2025-11-16 16:02:17 [INFO] RAGChatbot: Generated answer for: What is the PTO... -> Full-time employees...
2025-11-16 16:02:17 [INFO] RAGChatbot: ✅ Successfully answered: What is the PTO policy?
```

### Log Levels

- **DEBUG**: Detailed retrieval info, moderation checks
- **INFO**: Query processing, successful operations
- **WARNING**: Moderation flags, non-critical issues
- **ERROR**: API failures, missing files

### Monitoring Best Practices

- Check `logs/app.log` for errors
- Monitor moderation flags for abuse patterns
- Track query latency in production
- Set up alerts for API failures

## 🔒 Security

### Content Moderation

All queries pass through OpenAI Moderation API:

```python
from src.utils import moderate_content

if moderate_content(user_input):
    return {"error": "Unsafe content detected"}
```

**Flagged categories:**
- Hate speech
- Harassment
- Violence
- Self-harm
- Sexual content
- Dangerous content

### API Key Security

- ✅ Store API keys in `.env` (gitignored)
- ✅ Never commit `.env` to version control
- ✅ Use environment variables in production
- ✅ Rotate keys regularly

### Data Privacy

- No user data persisted beyond logs
- ChromaDB stores only document embeddings
- Logs can be sanitized before sharing

## 📈 Performance Characteristics

### Indexing Performance

- **Document size**: 3,742 characters
- **Chunks generated**: 23 chunks
- **Embedding calls**: 23 API calls
- **Index build time**: ~2-3 seconds

### Query Performance

- **Moderation check**: ~200ms
- **Vector search**: ~50ms (local)
- **LLM generation**: ~1-2 seconds
- **Total latency**: ~1.5-2.5 seconds

### Cost Estimation

**Per query:**
- Embedding: ~$0.00001 (1 embedding)
- LLM generation: ~$0.0001 (250 tokens)
- **Total**: ~$0.00011 per query

**Index build:**
- Embeddings: ~$0.00023 (23 chunks)
- One-time cost

## 🐛 Known Issues & Limitations

1. **Chunk Size Trade-off**: 300 chars may lose context for complex policies
2. **K=2 Limitation**: May miss relevant info if spread across >2 chunks
3. **Local ChromaDB**: Not suitable for distributed/production deployment
4. **No Caching**: Repeat queries re-call LLM (no response cache)
5. **Single Document**: System designed for one FAQ file

## 🚧 Future Improvements

- [ ] Add Streamlit web UI for easier interaction
- [ ] Implement response caching (Redis)
- [ ] Add metadata filtering (by department, policy type)
- [ ] Support multiple documents with source attribution
- [ ] Implement re-ranking for better retrieval
- [ ] Add conversation history for follow-up questions
- [ ] Deploy to cloud with managed vector DB (Pinecone/Weaviate)
- [ ] Add A/B testing framework for chunk size optimization
- [ ] Implement feedback loop for answer quality
- [ ] Add support for document updates without full rebuild

## 📄 License

This project is developed for educational purposes.

## 👤 Author

**Theophilus Adukpo**

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

This is an educational project. For questions or suggestions, please reach out to the author.

---

**Last Updated**: November 2025
