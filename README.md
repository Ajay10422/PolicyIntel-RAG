# PolicyIntel RAG

An end-to-end Retrieval-Augmented Generation (RAG) system for insurance and HR policy analysis, built to demonstrate Azure AI-102 competencies across six Azure AI services.

---

## What It Does

Upload insurance and HR policy PDFs. Ask natural-language questions. Get structured, cited answers with risk assessments, coverage gap analysis, and cross-document comparisons — all grounded in the actual policy text.

![Architecture](https://img.shields.io/badge/Azure-AI--102-0078D4?logo=microsoftazure&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

---

## Azure Services Used

| Service | Role | SDK |
|---|---|---|
| **Azure OpenAI** (o4-mini) | Structured JSON synthesis via `json_schema` response format | `openai` |
| **Azure OpenAI** (text-embedding-ada-002) | Vector embeddings for semantic search | `openai` |
| **Azure AI Search** | Hybrid search — BM25 + HNSW vector + semantic reranker | `azure-search-documents` |
| **Azure AI Document Intelligence** | Layout-aware PDF parsing (`prebuilt-layout` model) | `azure-ai-documentintelligence` |
| **Azure AI Language** | Key phrase extraction + Named Entity Recognition per chunk | `azure-ai-textanalytics` |
| **Azure AI Content Safety** | Input query and output response moderation | `azure-ai-contentsafety` |
| **Azure Monitor / App Insights** | Request telemetry, latency tracking, error logging | `azure-monitor-opentelemetry` |

---

## Architecture

```
PDF Documents
      │
      ▼
Hybrid_indexer.py
  ├─ Azure AI Document Intelligence  →  layout-aware paragraph extraction
  ├─ pypdf fallback                  →  used when DI token density < 150/page
  ├─ tiktoken chunker                →  800-token chunks with 100-token overlap
  ├─ Azure AI Language               →  key phrases + NER per chunk
  ├─ Azure OpenAI Embeddings         →  ada-002 vector per chunk
  └─ Azure AI Search upload          →  HNSW index with semantic config

User Query
      │
      ▼
main.py  (FastAPI)
  ├─ Azure AI Content Safety         →  input moderation (blocks severity ≥ 2)
  ├─ Azure OpenAI Embeddings         →  query vector
  ├─ Azure AI Search                 →  hybrid search + semantic reranker
  │    ├─ BM25 keyword match
  │    ├─ HNSW vector match
  │    └─ semantic reranker score
  ├─ Deduplication                   →  content-hash dedup across results
  ├─ Azure OpenAI o4-mini            →  structured JSON via json_schema mode
  ├─ Azure AI Content Safety         →  output moderation (non-blocking)
  └─ Structured Answer               →  summary · findings · gaps · risk · comparison

      │
      ▼
app.py  (Streamlit)
  ├─ Risk level pill  (Low / Medium / High)
  ├─ Answer markdown
  ├─ Citation cards   (source · page · reranker score · key phrases)
  └─ Structured Breakdown expander
```

<img width="5187" height="8191" alt="AWS Application Delivery-2026-04-05-161255" src="https://github.com/user-attachments/assets/dfbce327-63a6-4556-8aa4-959880c73987" />


---

## Features

- **Hybrid Search** — combines BM25 keyword matching with HNSW vector search, re-ranked by Azure's semantic reranker
- **Structured Output** — o4-mini returns typed JSON (`summary`, `key_findings`, `coverage_gaps`, `risk_level`, `cross_document_comparison`) enforced by a strict JSON schema
- **Page-level Citations** — every source card shows the exact page number and semantic reranker score
- **Mode-aware Depth** — Fast (5 sources, 3 findings) / Balanced (10 sources, 5 findings) / Deep (20 sources, 8 findings)
- **Content Safety** — input and output moderation via Azure AI Content Safety; fails open so a missing service never breaks queries
- **Responsible AI** — `DefaultAzureCredential` fallback when API keys are absent; explicit fail-open design on every optional service
- **Live Sidebar Stats** — real latency, top reranker score, and query count update after each response

---

## Project Structure

```
PolicyIntel-RAG/
├── app.py               # Streamlit frontend
├── main.py              # FastAPI backend — /ask endpoint
├── Hybrid_indexer.py    # One-time indexing pipeline
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template (copy to .env)
└── Data/                # Place your PDF policy documents here (not tracked)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Ajay10422/PolicyIntel-RAG.git
cd PolicyIntel-RAG
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your Azure service keys in .env
```

### 3. Provision Azure services

| Service | Portal path | Free tier |
|---|---|---|
| Azure OpenAI | AI Foundry → Deploy `o4-mini` + `text-embedding-ada-002` | — |
| Azure AI Search | Create resource → Standard S1 | — |
| Document Intelligence | Create resource | F0 (500 pages/month) |
| Language service | Create resource | F0 (5,000 records/month) |
| Content Safety | Create resource | F0 (5,000 records/month) |
| Application Insights | Create resource → Workspace-based | Free tier |

### 4. Add your PDFs

Place PDF policy documents in a `Data/` folder at the project root.

### 5. Index documents

```bash
python Hybrid_indexer.py
```

### 6. Run

```bash
# Terminal 1 — backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — frontend
streamlit run app.py
```

Open **http://localhost:8501**

---

## Example Queries

```
Compare the death benefit structure in the Principal life insurance policy
with the AD&D and liability coverages in the renters policy.

What personal property is excluded from Coverage C in the renters insurance policy?

What happens if I become disabled — compare what the employee handbook says
about short-term disability with what the life insurance policy covers?
```

---

## Key Design Decisions

**Chunk size: 800 tokens**
Each chunk represents roughly one page of content. Larger chunks (tested at 3,000) caused the table-of-contents page to dominate every query since it mentions all coverage names.

**DI quality check**
Document Intelligence is used only when it extracts ≥ 150 tokens per page on average. PDFs with multi-column or scanned layouts fall back to pypdf, which gives better text coverage on these document types.

**Deduplication**
Search results are deduplicated by content hash before building the LLM context. Without this, the same chunk can appear 5× in top-K results when the index has few large chunks.

**Strict JSON schema**
`cross_document_comparison` is typed as `string` (not `anyOf [string, null]`) to force the model to always return a comparison, not null. Per-field `description` keys in the schema act as additional per-field instructions to the model.

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. All Azure AI services except OpenAI and Search are optional — the system degrades gracefully when keys are absent.
