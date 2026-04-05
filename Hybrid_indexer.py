"""
Hybrid_indexer.py
=================
Policy RAG – Azure AI Search indexer with page_number tracking
"""

import os
import uuid
import traceback
from typing import List, Dict, Any

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Azure core
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Azure AI Search
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField, SearchFieldDataType, SearchableField,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
    SemanticSearch, SearchIndex, SearchField,
)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_configured(env_var: str) -> bool:
    val = os.getenv(env_var, "").strip()
    return bool(val) and "<" not in val


def get_azure_credential(key_env_var: str):
    key = os.getenv(key_env_var, "").strip()
    if key and "<" not in key:
        return AzureKeyCredential(key)
    return DefaultAzureCredential()


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------

_DOC_INTEL_CONFIGURED = _is_configured("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
_LANGUAGE_CONFIGURED = _is_configured("AZURE_LANGUAGE_ENDPOINT")

print(f"[Config] Document Intelligence: {'enabled' if _DOC_INTEL_CONFIGURED else 'not configured'}")
print(f"[Config] Azure AI Language:     {'enabled' if _LANGUAGE_CONFIGURED else 'not configured'}")

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_oa_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")

openai_client = OpenAI(
    base_url=f"{_oa_endpoint}/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    default_headers={"api-key": os.getenv("AZURE_OPENAI_KEY")},
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=get_azure_credential("AZURE_SEARCH_KEY"),
)

index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=get_azure_credential("AZURE_SEARCH_KEY"),
)

doc_intel_client = None
if _DOC_INTEL_CONFIGURED:
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        doc_intel_client = DocumentIntelligenceClient(
            endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            credential=get_azure_credential("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        )
    except Exception as e:
        print(f"[Config] Document Intelligence init failed: {e}")

language_client = None
if _LANGUAGE_CONFIGURED:
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        language_client = TextAnalyticsClient(
            endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT"),
            credential=get_azure_credential("AZURE_LANGUAGE_KEY"),
        )
    except Exception as e:
        print(f"[Config] Azure AI Language init failed: {e}")


# ---------------------------------------------------------------------------
# Index schema (with page_number)
# ---------------------------------------------------------------------------

def create_index() -> None:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="default"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),   # NEW
        SearchField(name="key_phrases", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True, filterable=True),
        SearchField(name="entities", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True, filterable=True),
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
        algorithms=[HnswAlgorithmConfiguration(name="default")],
    )

    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="key_phrases")],
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    try:
        index_client.create_or_update_index(index)
        print("Index created or updated successfully.")
    except Exception as exc:
        print(f"Index error: {exc}")
        raise


# ---------------------------------------------------------------------------
# PDF parsing with page tracking
# ---------------------------------------------------------------------------

def _pypdf_fallback(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    paragraphs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            paragraphs.append({"content": text, "role": "", "page_number": page_num})
    print(f"  pypdf extracted {len(paragraphs)} pages from {os.path.basename(pdf_path)}")
    return paragraphs


def analyze_pdf_with_document_intelligence(pdf_path: str) -> List[Dict[str, Any]]:
    if doc_intel_client is None:
        return _pypdf_fallback(pdf_path)

    # Count pages via pypdf to set a quality baseline
    from pypdf import PdfReader as _PR
    page_count = len(_PR(pdf_path).pages)

    paragraphs: List[Dict[str, Any]] = []
    try:
        with open(pdf_path, "rb") as f:
            poller = doc_intel_client.begin_analyze_document(
                "prebuilt-layout", body=f, content_type="application/octet-stream"
            )
        result = poller.result()

        if result.paragraphs:
            for para in result.paragraphs:
                role = para.role if para.role else ""
                content = para.content.strip() if para.content else ""
                page_num = getattr(para, "page_number", 1) if hasattr(para, "page_number") else 1
                if content:
                    paragraphs.append({"content": content, "role": role, "page_number": page_num})

        if not paragraphs and result.pages:
            for page_num, page in enumerate(result.pages, start=1):
                page_text = " ".join(line.content for line in (page.lines or []) if line.content).strip()
                if page_text:
                    paragraphs.append({"content": page_text, "role": "", "page_number": page_num})

        # Quality check: a typical page has ~250 words (~330 tokens).
        # If DI extracted less than 150 tokens per page on average the PDF layout
        # is too complex for DI (multi-column, scanned, form-heavy) — pypdf covers it better.
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(enc.encode(p["content"])) for p in paragraphs)
        tokens_per_page = total_tokens / max(page_count, 1)
        if tokens_per_page < 150:
            print(f"  DI extracted only ~{int(tokens_per_page)} tokens/page — using pypdf for better coverage")
            return _pypdf_fallback(pdf_path)

        print(f"  Document Intelligence extracted {len(paragraphs)} paragraphs from {os.path.basename(pdf_path)}")
        return paragraphs

    except Exception as exc:
        print(f"  Document Intelligence error: {exc} — falling back to pypdf")
        return _pypdf_fallback(pdf_path)


# ---------------------------------------------------------------------------
# Chunking with page_number
# ---------------------------------------------------------------------------

_HEADING_ROLES = {"title", "sectionHeading", "heading"}

def chunk_paragraphs(paragraphs: List[Dict[str, Any]], chunk_size: int = 6000, overlap: int = 500) -> List[Dict[str, Any]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: List[Dict[str, Any]] = []
    buffer_tokens: List[int] = []
    current_page = 1

    def flush(buf: List[int], page: int) -> List[int]:
        if buf:
            chunks.append({"text": encoding.decode(buf), "page_number": page})
            return buf[-overlap:] if overlap > 0 else []
        return []

    for para in paragraphs:
        text = para.get("content", "").strip()
        role = para.get("role", "")
        page_num = para.get("page_number", current_page)

        if not text:
            continue

        current_page = page_num
        para_tokens = encoding.encode(text)

        upper = text.upper()
        is_major_section = (
            role in _HEADING_ROLES or
            any(kw in upper for kw in ["COVERAGE E", "PERSONAL LIABILITY", "SECTION II",
                                       "COVERAGE F", "COVERAGE G", "MEDICAL PAYMENTS",
                                       "DECLARATIONS", "INSURING AGREEMENT"])
        )

        if is_major_section and len(buffer_tokens) > 200:
            buffer_tokens = flush(buffer_tokens, current_page)

        buffer_tokens.extend(para_tokens)

        while len(buffer_tokens) >= chunk_size:
            chunk_slice = buffer_tokens[:chunk_size]
            chunks.append({"text": encoding.decode(chunk_slice), "page_number": current_page})
            buffer_tokens = buffer_tokens[chunk_size - overlap:]

    if buffer_tokens:
        chunks.append({"text": encoding.decode(buffer_tokens), "page_number": current_page})

    print(f"  Created {len(chunks)} chunks (max size ~{chunk_size} tokens)")
    return chunks


# ---------------------------------------------------------------------------
# Language enrichment
# ---------------------------------------------------------------------------

def enrich_chunks_with_language(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if language_client is None:
        return [{"text": c["text"], "key_phrases": [], "entities": [], "page_number": c["page_number"]} for c in chunks]

    BATCH_SIZE = 5
    enriched: List[Dict[str, Any]] = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = [c["text"] for c in chunks[batch_start: batch_start + BATCH_SIZE]]

        kp_results = ner_results = None
        try:
            kp_results = language_client.extract_key_phrases(batch)
        except Exception as exc:
            print(f"  Key phrase error (batch {batch_start}): {exc}")

        try:
            ner_results = language_client.recognize_entities(batch)
        except Exception as exc:
            print(f"  NER error (batch {batch_start}): {exc}")

        for idx, chunk_text in enumerate(batch):
            key_phrases = []
            entities = []

            if kp_results and idx < len(kp_results):
                doc = kp_results[idx]
                if not doc.is_error:
                    key_phrases = list(doc.key_phrases)[:20]

            if ner_results and idx < len(ner_results):
                doc = ner_results[idx]
                if not doc.is_error:
                    for ent in doc.entities:
                        if ent.confidence_score > 0.8:
                            entities.append(ent.text)
                    seen = set()
                    entities = [e for e in entities if not (e in seen or seen.add(e))][:15]

            enriched.append({
                "text": chunk_text,
                "key_phrases": key_phrases,
                "entities": entities,
                "page_number": chunks[batch_start + idx]["page_number"]
            })

    return enriched


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        input=text[:8000],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Starting Policy RAG Indexing ===")
    create_index()

    data_folder = os.path.join(os.path.dirname(__file__), "Data")
    if not os.path.isdir(data_folder):
        data_folder = os.path.join(os.path.dirname(__file__), "data")

    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s) in {data_folder}\n")

    all_documents: List[Dict[str, Any]] = []

    for filename in pdf_files:
        pdf_path = os.path.join(data_folder, filename)
        print(f"Processing: {filename}")

        paragraphs = analyze_pdf_with_document_intelligence(pdf_path)
        if not paragraphs:
            print(f"  Skipping {filename} – no text extracted.\n")
            continue

        chunks = chunk_paragraphs(paragraphs, chunk_size=800, overlap=100)
        if not chunks:
            print(f"  No chunks produced for {filename}.\n")
            continue

        enriched = enrich_chunks_with_language(chunks)
        print(f"  Enriched {len(enriched)} chunk(s) with language metadata.")

        for chunk_idx, enriched_chunk in enumerate(enriched):
            chunk_text = enriched_chunk["text"]
            if not chunk_text.strip():
                continue

            print(f"  Embedding chunk {chunk_idx + 1}/{len(enriched)} …")
            try:
                embedding = get_embedding(chunk_text)
            except Exception as exc:
                print(f"  Embedding error for chunk {chunk_idx}: {exc}")
                continue

            all_documents.append({
                "id": str(uuid.uuid4()),
                "content": chunk_text,
                "embedding": embedding,
                "source": filename,
                "chunk_id": chunk_idx,
                "page_number": enriched_chunk.get("page_number", 1),
                "key_phrases": enriched_chunk["key_phrases"],
                "entities": enriched_chunk["entities"],
            })

        print(f"  Done: {filename}\n")

    if not all_documents:
        print("No documents to upload.")
        return

    UPLOAD_BATCH = 100
    total_uploaded = 0
    for i in range(0, len(all_documents), UPLOAD_BATCH):
        batch = all_documents[i: i + UPLOAD_BATCH]
        try:
            results = search_client.upload_documents(batch)
            total_uploaded += len(results)
            print(f"Uploaded batch {i // UPLOAD_BATCH + 1}: {len(results)} documents.")
        except Exception as exc:
            print(f"Upload error for batch {i}: {exc}")

    print(f"\nIndexing complete. {total_uploaded} document(s) uploaded.")


if __name__ == "__main__":
    main()