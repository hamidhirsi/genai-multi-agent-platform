# Architecture Deep Dive

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                   │
│  React + TypeScript Frontend (Vite, shadcn/ui, Tailwind CSS)           │
│  - SSE Event Streaming                                                   │
│  - Real-time Document Canvas                                             │
│  - Chat Interface with Citations                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HTTPS/WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API Gateway Layer                                │
│  FastAPI + Uvicorn (Python 3.11+)                                       │
│  - REST API (CRUD operations)                                            │
│  - SSE Streaming Endpoints                                               │
│  - JWT Authentication                                                     │
│  - CORS Configuration                                                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Domain Services Layer                               │
│  Pure Business Logic (No Infrastructure Dependencies)                    │
│                                                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐                      │
│  │ Assistant Service   │  │ Firm Knowledge      │                      │
│  │ - AI Chat           │  │ - Document Upload   │                      │
│  │ - Function Calling  │  │ - Semantic Search   │                      │
│  │ - Context Building  │  │ - Folder Mgmt       │                      │
│  └─────────────────────┘  └─────────────────────┘                      │
│                                                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐                      │
│  │ Case Management     │  │ Workflow Engine     │                      │
│  │ - Classification    │  │ - Template Registry │                      │
│  │ - QA Assessment     │  │ - Step Execution    │                      │
│  │ - Draft Generation  │  │ - Error Handling    │                      │
│  └─────────────────────┘  └─────────────────────┘                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ Repository Interfaces
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                                  │
│  External System Integrations (Injected at Runtime)                     │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              AI Orchestration                             │          │
│  │  ┌────────────────┐    ┌────────────────┐                │          │
│  │  │ LangGraph      │    │ OpenAI Client  │                │          │
│  │  │ - StateGraph   │    │ - Chat API     │                │          │
│  │  │ - MemorySaver  │    │ - Embeddings   │                │          │
│  │  │ - Checkpointer │    │ - Function Call│                │          │
│  │  └────────────────┘    └────────────────┘                │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              Document Processing                          │          │
│  │  ┌────────────────┐    ┌────────────────┐                │          │
│  │  │ Text Extractor │    │ Chunking       │                │          │
│  │  │ - PyPDF2       │    │ - Section-aware│                │          │
│  │  │ - python-docx  │    │ - 800 tokens   │                │          │
│  │  │ - Tesseract OCR│    │ - 100 overlap  │                │          │
│  │  └────────────────┘    └────────────────┘                │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              Persistence Layer                            │          │
│  │  ┌────────────────┐    ┌────────────────┐                │          │
│  │  │ Qdrant Repo    │    │ PostgreSQL Repo│                │          │
│  │  │ - Per-tenant   │    │ - tenant_id idx│                │          │
│  │  │ - Vector search│    │ - JSONB columns│                │          │
│  │  │ - Chunk storage│    │ - Relationships│                │          │
│  │  └────────────────┘    └────────────────┘                │          │
│  └──────────────────────────────────────────────────────────┘          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Storage Layer                                │
│                                                                           │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐ │
│  │   PostgreSQL 14+   │  │   Qdrant 1.7+      │  │  Local Storage   │ │
│  │                    │  │                    │  │                  │ │
│  │ - Cases            │  │ - legal_knowledge  │  │ - Uploaded files │ │
│  │ - Chat Messages    │  │ - firm_knowledge_* │  │ - Generated PDFs │ │
│  │ - Users/Firms      │  │   (per tenant)     │  │ - Temp files     │ │
│  │ - Timeline Entries │  │                    │  │                  │ │
│  │ - Knowledge Docs   │  │ Collections:       │  │ Directory:       │ │
│  │   (metadata)       │  │ - Vectors (1536d)  │  │ data/storage/    │ │
│  │                    │  │ - Payloads         │  │                  │ │
│  └────────────────────┘  └────────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Observability Layer                                 │
│                                                                           │
│  ┌────────────────────┐  ┌────────────────────┐                        │
│  │  Langfuse Cloud    │  │  Application Logs  │                        │
│  │  - Traces          │  │  - Python logging  │                        │
│  │  - Spans           │  │  - JSON format     │                        │
│  │  - Sessions        │  │  - Structured logs │                        │
│  │  - Token Usage     │  │                    │                        │
│  └────────────────────┘  └────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph Workflow State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                     Case Intake Workflow                         │
└─────────────────────────────────────────────────────────────────┘

                           START
                             │
                             ▼
                    ┌─────────────────┐
                    │  Intake Node    │
                    │                 │
                    │ - Validate      │
                    │   required      │
                    │   fields        │
                    │ - Check email   │
                    │   format        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Classify Node   │
                    │                 │
                    │ - Matter types  │
                    │ - Urgency level │
                    │ - Jurisdiction  │
                    │ - Confidence    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ QA Assess Node  │
                    │                 │
                    │ - Completeness  │
                    │ - Missing info  │
                    │ - Sufficiency   │
                    │   score         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Create Case Node│
                    │                 │
                    │ - Save to DB    │
                    │ - Timeline      │
                    │ - Assign status │
                    │ - Set priority  │
                    └────────┬────────┘
                             │
                             ▼
                           END

State Shape (TypedDict):
{
  case_id: str
  case_data: Dict[str, Any]
  classification: Optional[Dict]
  qa_assessment: Optional[Dict]
  current_step: str
  error_messages: List[str]
  step_timings: Dict[str, float]
  processing_complete: bool
}
```

---

## RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Document Ingestion Pipeline                         │
└─────────────────────────────────────────────────────────────────┘

User Upload (PDF/DOCX/Image)
          │
          ▼
┌──────────────────────┐
│  Text Extraction     │
│                      │
│  - PyPDF2 (primary)  │
│  - pdfplumber        │
│    (fallback)        │
│  - python-docx       │
│  - Tesseract OCR     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Semantic Chunking    │
│                      │
│ 1. Detect sections   │
│    (headers, lists)  │
│                      │
│ 2. Split sentences   │
│    (smart patterns)  │
│                      │
│ 3. Create chunks     │
│    (800 tokens)      │
│                      │
│ 4. Add overlap       │
│    (100 tokens)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Embedding Generation │
│                      │
│ OpenAI API:          │
│ text-embedding-      │
│ ada-002              │
│                      │
│ Output: 1536D vector │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Qdrant Storage       │
│                      │
│ Collection:          │
│ firm_knowledge_      │
│ {firm_id}            │
│                      │
│ Payload:             │
│ - chunk_content      │
│ - chunk_index        │
│ - section_title      │
│ - document_id        │
│ - metadata           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ PostgreSQL Metadata  │
│                      │
│ knowledge_documents: │
│ - id (UUID)          │
│ - firm_id            │
│ - folder_id          │
│ - filename           │
│ - storage_path       │
│ - extracted_text     │
│ - vector_id          │
│ - tags[]             │
└──────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              Retrieval & Re-Ranking Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

User Query: "pregnancy discrimination case law"
          │
          ▼
┌───────────────────────────────────────────────────┐
│  Dual Source Search (Parallel)                    │
│                                                    │
│  ┌──────────────────┐    ┌──────────────────┐   │
│  │ Statutory Law    │    │ Firm Knowledge   │   │
│  │ (legal_knowledge)│    │ (firm_knowledge_ │   │
│  │                  │    │  {firm_id})      │   │
│  │ Limit: 10        │    │ Limit: 10        │   │
│  └────────┬─────────┘    └─────────┬────────┘   │
│           │                         │             │
│           └────────┬────────────────┘             │
└────────────────────┼──────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Weighted Re-Ranking │
          │                      │
          │  Statutory: 1.0x     │
          │  Firm: 1.5x boost    │
          │                      │
          │  Sort by weighted    │
          │  score descending    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Top-K Selection     │
          │                      │
          │  Return top 5        │
          │  regardless of       │
          │  source type         │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Format Response     │
          │                      │
          │  {                   │
          │    "sources": [...], │
          │    "statutory_count":│
          │    "firm_count":     │
          │    "context": "..."  │
          │  }                   │
          └──────────────────────┘
```

---

## Multi-Tenant Data Isolation

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-Tenant Architecture                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  API Request    │
                    │  (with JWT)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Extract firm_id │
                    │ from JWT claims │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
            ▼                                  ▼
  ┌──────────────────┐            ┌──────────────────┐
  │   PostgreSQL     │            │     Qdrant       │
  │                  │            │                  │
  │ Query Pattern:   │            │ Collection Name: │
  │ SELECT * FROM    │            │ firm_knowledge_  │
  │   cases          │            │ {firm_id}        │
  │ WHERE            │            │                  │
  │   tenant_id =    │            │ Complete         │
  │   'firm_123'     │            │ isolation        │
  │   ^^^^^^^^^^^^^^ │            │ per tenant       │
  │   INDEXED!       │            │                  │
  │                  │            │                  │
  │ Benefits:        │            │ Benefits:        │
  │ - Row-level      │            │ - No filtering   │
  │   security       │            │   needed         │
  │ - Fast lookups   │            │ - Drop entire    │
  │ - JSONB for AI   │            │   collection     │
  │   results        │            │ - Independent    │
  └──────────────────┘            │   scaling        │
                                  └──────────────────┘

Example Qdrant Collections:
- legal_knowledge_v2           (Shared statutory law)
- firm_knowledge_default-firm  (Default tenant)
- firm_knowledge_acme_law      (Acme Law Firm)
- firm_knowledge_smith_partners (Smith & Partners)


Repository Factory Pattern:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  RepositoryFactory                                           │
│  ───────────────────                                         │
│                                                              │
│  create_document_repository(firm_id: str)                   │
│      │                                                       │
│      └─> Returns: QdrantDocumentRepository                  │
│            - Scoped to firm_id                              │
│            - Collection: firm_knowledge_{firm_id}           │
│            - All operations isolated                        │
│                                                              │
│  create_case_repository(firm_id: str)                       │
│      │                                                       │
│      └─> Returns: PostgresCaseRepository                    │
│            - Auto-adds tenant_id filter                     │
│            - All queries scoped                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## OpenAI Function Calling Flow

```
User Message: "Search for unfair dismissal cases"
          │
          ▼
┌──────────────────────┐
│ Build Chat Messages  │
│                      │
│ - System prompt      │
│ - Conversation hist  │
│ - User message       │
│ - Case context       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ OpenAI Chat API      │
│                      │
│ model: gpt-4o        │
│ tools: [             │
│   search_legal_...   │
│   generate_letter... │
│   update_document... │
│   analyze_timeline...│
│   assess_risks...    │
│   execute_workflow...│
│   list_workflows...  │
│ ]                    │
│ tool_choice: "auto"  │
│ stream: true         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ AI Decides to Call   │
│ Function             │
│                      │
│ {                    │
│   "name": "search_   │
│     legal_knowledge",│
│   "arguments": {     │
│     "query": "unfair │
│       dismissal",    │
│     "top_k": 5       │
│   }                  │
│ }                    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Execute Function     │
│                      │
│ 1. Statutory search  │
│ 2. Firm search       │
│ 3. Weighted re-rank  │
│ 4. Return sources    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Add Function Result  │
│ to Conversation      │
│                      │
│ [                    │
│   {role: "assistant",│
│    tool_calls: [...]},│
│   {role: "tool",     │
│    content: {...}}   │
│ ]                    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Follow-up OpenAI     │
│ Call with Results    │
│                      │
│ AI generates response│
│ using retrieved      │
│ sources with         │
│ citations            │
└──────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Production Deployment                      │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Load Balancer │
                    │   (Nginx/Traefik)│
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
            ▼                                  ▼
  ┌──────────────────┐            ┌──────────────────┐
  │  Frontend        │            │   Backend API    │
  │  (React SPA)     │            │   (FastAPI)      │
  │                  │            │                  │
  │  Nginx static    │            │  Uvicorn workers │
  │  file server     │            │  (4-8 workers)   │
  │                  │            │                  │
  │  Port: 80/443    │            │  Port: 8000      │
  └──────────────────┘            └────────┬─────────┘
                                           │
                         ┌─────────────────┼─────────────────┐
                         │                 │                 │
                         ▼                 ▼                 ▼
              ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
              │  PostgreSQL   │ │    Qdrant     │ │   OpenAI API  │
              │               │ │               │ │               │
              │  - Persistent │ │  - Persistent │ │  - Managed    │
              │    volume     │ │    volume     │ │    service    │
              │  - Connection │ │  - 6333, 6334 │ │               │
              │    pool (20)  │ │    ports      │ │               │
              └───────────────┘ └───────────────┘ └───────────────┘

Docker Compose Services:
- postgres:14
- qdrant/qdrant:v1.7.4
- backend:latest (custom image)
- frontend:latest (custom image)

Volumes:
- postgres_data:/var/lib/postgresql/data
- qdrant_data:/qdrant/storage
- backend_storage:/app/data/storage

Environment:
- Production: .env.production
- Staging: .env.staging
- Development: .env.local
```

---

## Observability Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Langfuse Observability                        │
└─────────────────────────────────────────────────────────────────┘

Application Events
       │
       ▼
┌──────────────────┐
│ LangfuseTracer   │
│ (Custom Wrapper) │
│                  │
│ - trace_agent    │
│   _call()        │
│ - create_span()  │
│ - trace_         │
│   conversation() │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Langfuse SDK     │
│                  │
│ - Start span     │
│ - Update output  │
│ - End span       │
│ - Flush buffer   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Langfuse Cloud   │
│                  │
│ Dashboard views: │
│                  │
│ • Traces         │
│   - Nested spans │
│   - Latency tree │
│   - Error traces │
│                  │
│ • Sessions       │
│   - By case_id   │
│   - User journey │
│   - Conversation │
│     flow         │
│                  │
│ • Metrics        │
│   - Token usage  │
│   - Costs        │
│   - Latency p50/ │
│     p95/p99      │
│   - Error rate   │
│                  │
│ • Datasets       │
│   - Test cases   │
│   - Evaluations  │
└──────────────────┘


Traced Operations:
┌────────────────────────────────────────────────────────┐
│ Operation             │ Span Name        │ Metadata    │
├────────────────────────────────────────────────────────┤
│ AI Chat               │ openai_chat      │ model, msg  │
│ Function Call         │ function_{name}  │ args, output│
│ Workflow Step         │ workflow_step_*  │ duration_ms │
│ RAG Search            │ rag_search       │ sources_cnt │
│ Classification        │ classification   │ confidence  │
│ QA Assessment         │ qa_assessment    │ score       │
└────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Security Layers                           │
└─────────────────────────────────────────────────────────────────┘

1. Authentication & Authorization
   ┌──────────────────────────────────────┐
   │ JWT Token (Header: Authorization)    │
   │                                       │
   │ Payload:                              │
   │ {                                     │
   │   "sub": "user_id",                   │
   │   "firm_id": "firm_123",              │
   │   "role": "lawyer",                   │
   │   "exp": 1234567890                   │
   │ }                                     │
   │                                       │
   │ Verification:                         │
   │ - Signature check (HS256)             │
   │ - Expiration validation               │
   │ - Role-based access control           │
   └──────────────────────────────────────┘

2. Multi-Tenant Isolation
   ┌──────────────────────────────────────┐
   │ Request → Extract firm_id → Scope    │
   │                                       │
   │ PostgreSQL:                           │
   │ WHERE tenant_id = extracted_firm_id   │
   │                                       │
   │ Qdrant:                               │
   │ Collection = firm_knowledge_{firm_id} │
   │                                       │
   │ File Storage:                         │
   │ Path = data/storage/{firm_id}/...     │
   └──────────────────────────────────────┘

3. Input Validation
   ┌──────────────────────────────────────┐
   │ Pydantic Models:                      │
   │ - Type checking                       │
   │ - Length limits                       │
   │ - Email validation                    │
   │ - Enum constraints                    │
   │                                       │
   │ SQL Injection Protection:             │
   │ - SQLAlchemy ORM (parameterized)      │
   │ - No raw SQL queries                  │
   └──────────────────────────────────────┘

4. LLM Security
   ┌──────────────────────────────────────┐
   │ Prompt Injection Defense:             │
   │ - System prompt guards                │
   │ - User input sanitization             │
   │ - Output validation                   │
   │                                       │
   │ Content Filtering:                    │
   │ - OpenAI moderation API               │
   │ - PII redaction in logs               │
   │ - Rate limiting (100 req/min)         │
   └──────────────────────────────────────┘

5. Data Protection
   ┌──────────────────────────────────────┐
   │ At Rest:                              │
   │ - PostgreSQL encryption               │
   │ - File system permissions             │
   │                                       │
   │ In Transit:                           │
   │ - HTTPS/TLS 1.3                       │
   │ - Certificate pinning                 │
   │                                       │
   │ Logging:                              │
   │ - PII masking                         │
   │ - Secure log storage                  │
   │ - Audit trails                        │
   └──────────────────────────────────────┘
```

---

## Technology Decision Matrix

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| **Vector DB** | Qdrant, Pinecone, Weaviate, Chroma | **Qdrant** | • Open-source<br>• Self-hosted option<br>• Fast search (<100ms)<br>• Python client<br>• Per-tenant collections |
| **Workflow Orchestration** | LangGraph, LangChain, Temporal, Airflow | **LangGraph** | • AI-native state machines<br>• Checkpointing built-in<br>• TypedDict state<br>• Conditional routing<br>• Integration with LangChain |
| **LLM Provider** | OpenAI, Anthropic, Google, Local (Ollama) | **OpenAI** | • Best function calling<br>• Fast response times<br>• Vision API support<br>• Reliable uptime<br>• (Future: LiteLLM multi-provider) |
| **Embeddings** | OpenAI ada-002, Sentence Transformers, Cohere | **OpenAI ada-002** | • 1536 dimensions<br>• State-of-the-art quality<br>• Cost-effective ($0.0001/1K tokens)<br>• Same vendor as LLM |
| **Observability** | Langfuse, LangSmith, Weights & Biases, Arize | **Langfuse** | • LLM-specific traces<br>• Session analytics<br>• Token usage tracking<br>• Open-source option<br>• Active development |
| **Web Framework** | FastAPI, Flask, Django, Express | **FastAPI** | • Async/await native<br>• Auto OpenAPI docs<br>• Pydantic validation<br>• SSE streaming<br>• Type hints support |
| **Database** | PostgreSQL, MySQL, MongoDB | **PostgreSQL** | • JSONB for AI results<br>• Full-text search<br>• Strong ACID guarantees<br>• Mature ecosystem<br>• Multi-tenant patterns |

---

## File Structure & Layers

```
backend/
│
├── api/                      # API Layer (FastAPI routes)
│   └── v1/
│       ├── assistant.py      # SSE streaming, chat endpoint
│       ├── knowledge.py      # Document upload/search
│       ├── workflows.py      # Workflow triggers
│       └── cases.py          # CRUD operations
│
├── domain/                   # Business Logic Layer (pure Python)
│   ├── assistant/
│   │   ├── service.py        # Chat orchestration, function calling
│   │   └── function_registry.py
│   ├── firm_knowledge/
│   │   └── service.py        # Document management
│   ├── workflow_engine/
│   │   ├── executor.py       # Workflow execution logic
│   │   ├── templates/        # Workflow definitions
│   │   └── models.py         # DTOs, value objects
│   └── shared/
│       ├── models.py         # Shared domain models
│       └── repository.py     # Repository interfaces
│
├── infrastructure/           # External Integrations Layer
│   ├── ai/
│   │   └── embedding_provider.py
│   ├── document/
│   │   ├── chunking_strategy.py
│   │   └── text_extractor.py
│   ├── persistence/
│   │   ├── qdrant_document_repository.py
│   │   ├── postgres_case_repository.py
│   │   └── repository_factory.py
│   └── agents/
│       └── ollama_*.py
│
├── core/                     # Cross-cutting Concerns
│   ├── observability/
│   │   └── langfuse_tracer.py
│   └── workflow/
│       ├── case_workflow.py  # LangGraph state machine
│       └── state.py          # TypedDict definitions
│
├── models/                   # Data Models Layer (SQLAlchemy)
│   ├── case.py
│   ├── knowledge.py
│   └── database.py
│
└── services/                 # Legacy Service Layer
    ├── rag_service.py
    ├── embedding_service.py
    └── text_extraction_service.py
```

**Dependency Flow:**
```
API Layer
   ↓ calls
Domain Layer (pure business logic)
   ↓ uses interfaces from
Repository Interfaces
   ↑ implemented by
Infrastructure Layer (external systems)
   ↓ persists to
Data Models & Storage
```

**Key Principles:**
- **Domain Layer**: No infrastructure dependencies (testable, portable)
- **Infrastructure Layer**: Implements repository interfaces (swappable)
- **API Layer**: Thin controllers, delegates to domain services
- **Core Layer**: Cross-cutting concerns (observability, shared utilities)

---

## Performance Optimization Strategies

### 1. Connection Pooling
```python
# PostgreSQL connection pool
DATABASE_URL = "postgresql://user:pass@host/db?pool_size=20&max_overflow=10"

# Qdrant client reuse (singleton pattern)
qdrant_client = QdrantClient(url=settings.QDRANT_URL)
```

### 2. Batch Operations
```python
# Batch embedding generation
embeddings = await openai.embeddings.create(
    input=[chunk.content for chunk in chunks],  # Batch of 50
    model="text-embedding-ada-002"
)

# Batch Qdrant upsert
qdrant.upsert(collection_name=collection, points=points)  # 100 points
```

### 3. Async Operations
```python
# Parallel knowledge source search
statutory_task = asyncio.create_task(
    legal_research.search_legal_knowledge(query, limit=10)
)
firm_task = asyncio.create_task(
    firm_service.search_knowledge(query, limit=10)
)
statutory_results, firm_results = await asyncio.gather(statutory_task, firm_task)
```

### 4. Chunk-Level Search with Document Re-Ranking
```python
# Search 3x limit at chunk level
chunk_results = qdrant.search(query_vector, limit=limit * 3)

# Group by document_id and re-rank
documents = group_and_rerank(chunk_results)
return documents[:limit]  # Top-K documents
```

### 5. SSE Streaming (Perceived Performance)
```python
# Stream tokens immediately (don't wait for full response)
async for chunk in openai_stream:
    if chunk.choices[0].delta.content:
        yield {"type": "text", "content": chunk.choices[0].delta.content}
```

---

**Last Updated**: February 2025
