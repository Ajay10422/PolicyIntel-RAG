"""
main.py
=======
Policy RAG – FastAPI backend.

Endpoints:
  POST /ask   – accepts a natural-language query, runs hybrid semantic search
                against Azure AI Search, calls Azure OpenAI o4-mini for
                structured synthesis, and returns a rich Answer payload.

Features:
  - get_azure_credential() helper (key → managed identity fallback)
  - Optional Application Insights telemetry via azure-monitor-opentelemetry
  - Azure AI Content Safety checks on both input query and generated answer
  - Hybrid + semantic search with reranker scores and keyword highlights
  - Structured JSON output via response_format (o4-mini JSON schema mode)
  - Pydantic models: Citation, StructuredAnswer, Answer
"""

import os
import json
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Azure core
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Azure AI Search
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Azure AI Content Safety
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions

# Azure OpenAI
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Optional: Application Insights telemetry
# ---------------------------------------------------------------------------
_ai_conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
if _ai_conn_str and not _ai_conn_str.startswith("<"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=_ai_conn_str)
        print("Azure Monitor / Application Insights telemetry configured.")
    except Exception as _monitor_exc:
        print(f"Application Insights setup skipped: {_monitor_exc}")

# ---------------------------------------------------------------------------
# Credential helper
# ---------------------------------------------------------------------------

def get_azure_credential(key_env_var: str):
    """
    Returns AzureKeyCredential when the env var contains a real key value.
    Falls back to DefaultAzureCredential for managed-identity auth.
    """
    key = os.getenv(key_env_var, "").strip()
    if key and not key.startswith("<"):
        return AzureKeyCredential(key)
    return DefaultAzureCredential()


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

# Azure OpenAI (v1 SDK, Foundry endpoint)
_oa_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
openai_client = OpenAI(
    base_url=f"{_oa_endpoint}/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    default_headers={"api-key": os.getenv("AZURE_OPENAI_KEY")},
)

# Azure AI Search
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=get_azure_credential("AZURE_SEARCH_KEY"),
)

# Azure AI Content Safety
_cs_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", "").strip()
content_safety_client: Optional[ContentSafetyClient] = None
if _cs_endpoint and not _cs_endpoint.startswith("<"):
    try:
        content_safety_client = ContentSafetyClient(
            endpoint=_cs_endpoint,
            credential=get_azure_credential("AZURE_CONTENT_SAFETY_KEY"),
        )
    except Exception as _cs_exc:
        print(f"Content Safety client init skipped: {_cs_exc}")

# ---------------------------------------------------------------------------
# Structured output schema (o4-mini JSON schema mode)
# ---------------------------------------------------------------------------

ANSWER_SCHEMA = {
    "name": "policy_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A direct 2-4 sentence answer to the user's question. "
                    "Lead with the most important fact. "
                    "Do not begin with 'I' or restate the question."
                ),
            },
            "key_findings": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": (
                        "One specific, verifiable finding from the policy documents. "
                        "Always include the source filename and page number in parentheses. "
                        "Example: 'Death benefit is $10,000, reduced to 65% at ages 70-74 "
                        "(Principal-Sample-Life-Insurance-Policy.pdf, Part IV Section A).'"
                    ),
                },
                "description": "Concrete findings extracted directly from the policy text.",
            },
            "coverage_gaps": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": (
                        "A specific gap, ambiguity, exclusion, or limitation identified. "
                        "State what is missing or unclear and why it matters to the insured."
                    ),
                },
                "description": "Gaps, exclusions, or ambiguities that could leave the insured exposed.",
            },
            "risk_level": {
                "type": "string",
                "enum": ["Low", "Medium", "High"],
                "description": (
                    "Overall compliance and coverage risk. "
                    "Low = clear coverage, no significant gaps. "
                    "Medium = some ambiguity or minor gaps present. "
                    "High = significant exclusions, missing coverage, or legal exposure."
                ),
            },
            "cross_document_comparison": {
                "type": "string",
                "description": (
                    "REQUIRED — never omit this field. "
                    "When multiple documents are cited: directly compare how each document "
                    "handles the topic (e.g. what one covers that another does not). "
                    "When only one document is cited: compare the policy's approach to "
                    "standard industry practice or typical policyholder expectations. "
                    "Minimum one substantive sentence."
                ),
            },
        },
        "required": [
            "summary",
            "key_findings",
            "coverage_gaps",
            "risk_level",
            "cross_document_comparison",
        ],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    source: str
    content: str
    page_number: Optional[int] = None
    reranker_score: Optional[float] = None
    key_phrases: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    highlights: Optional[str] = None


class StructuredAnswer(BaseModel):
    summary: str
    key_findings: List[str]
    coverage_gaps: List[str]
    risk_level: str  # "Low" | "Medium" | "High"
    cross_document_comparison: Optional[str] = None


class Question(BaseModel):
    query: str
    top_k: Optional[int] = 10
    mode: Optional[str] = "Balanced"


class Answer(BaseModel):
    answer: str
    citations: List[Citation]
    structured: Optional[StructuredAnswer] = None
    safety_flagged: bool = False


# ---------------------------------------------------------------------------
# Content Safety helper
# ---------------------------------------------------------------------------

def check_content_safety(text: str) -> dict:
    """
    Analyses the first 1000 characters of *text* against Azure AI Content Safety.

    Returns {"safe": bool, "categories": list}.
    Fails open (returns safe=True) when the client is unavailable or an
    exception occurs, so a missing / unconfigured service never blocks queries.
    """
    if content_safety_client is None:
        return {"safe": True, "categories": []}

    try:
        response = content_safety_client.analyze_text(
            AnalyzeTextOptions(text=text[:1000])
        )
        flagged_categories: List[str] = []
        for category_result in response.categories_analysis:
            if category_result.severity >= 2:
                flagged_categories.append(
                    f"{category_result.category}(severity={category_result.severity})"
                )
        return {
            "safe": len(flagged_categories) == 0,
            "categories": flagged_categories,
        }
    except Exception as exc:
        print(f"Content Safety check error (failing open): {exc}")
        return {"safe": True, "categories": []}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="PolicyIntel RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Main RAG endpoint.

    Flow:
      1. Content safety check on input query
      2. Generate embedding
      3. Hybrid + semantic search (with highlights)
      4. Build citations (reranker score, highlights, key phrases, entities)
      5. LLM structured synthesis via o4-mini JSON schema mode
      6. Parse structured response
      7. Build formatted markdown answer
      8. Content safety check on output (non-blocking)
      9. Return Answer
    """
    # ------------------------------------------------------------------
    # Mode configuration — controls search depth and LLM verbosity
    # ------------------------------------------------------------------
    MODE_CONFIG = {
        "Fast": {
            "top_k":       5,
            "system_note": (
                "Be concise. Provide a 1-2 sentence summary. "
                "List exactly 3 key findings and 2 coverage gaps maximum."
            ),
        },
        "Balanced": {
            "top_k":       question.top_k,
            "system_note": (
                "Provide a thorough analysis. "
                "List 4-6 key findings and 3-4 coverage gaps."
            ),
        },
        "Deep": {
            "top_k":       max(question.top_k, 20),
            "system_note": (
                "Perform an exhaustive review of every clause, condition, and exclusion. "
                "List 6-8 key findings and every gap you can identify. "
                "Cross-reference across all source documents."
            ),
        },
    }
    cfg = MODE_CONFIG.get(question.mode or "Balanced", MODE_CONFIG["Balanced"])
    effective_top_k = cfg["top_k"]

    try:
        # ------------------------------------------------------------------
        # Step 1: Content safety on the input query
        # ------------------------------------------------------------------
        input_safety = check_content_safety(question.query)
        if not input_safety["safe"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Query flagged by content safety: "
                    f"{', '.join(input_safety['categories'])}"
                ),
            )

        # ------------------------------------------------------------------
        # Step 2: Generate embedding
        # ------------------------------------------------------------------
        emb_response = openai_client.embeddings.create(
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
            input=question.query,
        )
        query_vector = emb_response.data[0].embedding

        # ------------------------------------------------------------------
        # Step 3: Hybrid + semantic search
        # ------------------------------------------------------------------
        v_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=effective_top_k,
            fields="embedding",
        )

        # Fetch 3x more than needed so deduplication leaves enough unique chunks.
        fetch_top = max(effective_top_k * 3, 20)
        raw_results = list(search_client.search(
            search_text=question.query,
            vector_queries=[v_query],
            select=["content", "source", "page_number", "key_phrases", "entities"],
            top=fetch_top,
            query_type="semantic",
            semantic_configuration_name="default",
            highlight_fields="content",
        ))

        # Deduplicate by content hash — same chunk can score top-N multiple times
        seen: set = set()
        deduped = []
        for doc in raw_results:
            h = hash(doc.get("content", ""))
            if h not in seen:
                seen.add(h)
                deduped.append(doc)
            if len(deduped) >= effective_top_k:
                break

        # ------------------------------------------------------------------
        # Step 4: Build citations and context
        # ------------------------------------------------------------------
        context_list: List[str] = []
        citations: List[Citation] = []

        for doc in deduped:
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            page_number = doc.get("page_number")
            key_phrases = doc.get("key_phrases") or []
            entities = doc.get("entities") or []

            try:
                reranker_score = doc.get("@search.reranker_score")
            except Exception:
                reranker_score = None

            try:
                highlights_dict = doc.get("@search.highlights") or {}
                content_highlights = highlights_dict.get("content", [])
                highlight_text = " … ".join(content_highlights) if content_highlights else None
            except Exception:
                highlight_text = None

            context_list.append(f"SOURCE: {source} (page {page_number})\nTEXT: {content}")
            citations.append(
                Citation(
                    source=source,
                    content=content[:500] + ("..." if len(content) > 500 else ""),
                    page_number=page_number,
                    reranker_score=reranker_score,
                    key_phrases=key_phrases[:20] if key_phrases else [],
                    entities=entities[:15] if entities else [],
                    highlights=highlight_text,
                )
            )

        if not context_list:
            return Answer(
                answer="No relevant policy sections found for this query.",
                citations=[],
                structured=None,
                safety_flagged=False,
            )

        # ------------------------------------------------------------------
        # Step 5: LLM structured synthesis
        # ------------------------------------------------------------------
        system_prompt = (
            "You are a Senior HR and Insurance Policy Analyst with 20 years of experience "
            "reviewing group life, renters, and employee benefit policies. "
            "Your job is to extract precise, verifiable facts from the provided policy excerpts "
            "and return them in the structured JSON format defined by the schema. "
            "\n\nRules you must follow:\n"
            "1. Only cite facts that appear in the provided SOURCE text — do not invent coverage amounts or clause numbers.\n"
            "2. Every key_finding must reference the exact source filename and section or page.\n"
            "3. cross_document_comparison is MANDATORY — always populate it with a substantive comparison. Never return an empty string.\n"
            "4. coverage_gaps must describe the practical impact on the insured, not just restate what is absent.\n"
            "5. risk_level must reflect actual coverage exposure: if amounts are unspecified or exclusions are broad, rate Medium or High.\n"
            f"\nDepth mode: {cfg['system_note']}"
        )

        user_prompt = (
            f"Policy Context:\n\n{chr(10).join(context_list)}\n\n"
            f"User Question: {question.query}"
        )

        llm_response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
            response_format={"type": "json_schema", "json_schema": ANSWER_SCHEMA},
        )

        raw_json = llm_response.choices[0].message.content or "{}"

        # ------------------------------------------------------------------
        # Step 6: Parse structured response
        # ------------------------------------------------------------------
        structured: Optional[StructuredAnswer] = None
        try:
            parsed = json.loads(raw_json)
            structured = StructuredAnswer(
                summary=parsed.get("summary", ""),
                key_findings=parsed.get("key_findings", []),
                coverage_gaps=parsed.get("coverage_gaps", []),
                risk_level=parsed.get("risk_level", "Medium"),
                cross_document_comparison=parsed.get("cross_document_comparison"),
            )
        except Exception as parse_exc:
            print(f"Structured response parse error: {parse_exc}")
            # Graceful fallback: return raw text as summary
            structured = StructuredAnswer(
                summary=raw_json[:1000],
                key_findings=[],
                coverage_gaps=[],
                risk_level="Medium",
                cross_document_comparison=None,
            )

        # ------------------------------------------------------------------
        # Step 7: Build formatted markdown answer
        # ------------------------------------------------------------------
        risk_badge_map = {
            "Low": "🟢 **Risk Level: Low**",
            "Medium": "🟡 **Risk Level: Medium**",
            "High": "🔴 **Risk Level: High**",
        }
        risk_badge = risk_badge_map.get(structured.risk_level, f"**Risk Level: {structured.risk_level}**")

        answer_parts: List[str] = []

        # Summary (bold lead)
        answer_parts.append(f"**Summary**\n\n{structured.summary}")

        # Key findings
        if structured.key_findings:
            findings_md = "\n".join(f"- {f}" for f in structured.key_findings)
            answer_parts.append(f"**Key Findings**\n\n{findings_md}")

        # Coverage gaps
        if structured.coverage_gaps:
            gaps_md = "\n".join(f"- {g}" for g in structured.coverage_gaps)
            answer_parts.append(f"**Coverage Gaps**\n\n{gaps_md}")

        # Cross-document comparison
        if structured.cross_document_comparison:
            answer_parts.append(
                f"**Cross-Document Comparison**\n\n{structured.cross_document_comparison}"
            )

        # Risk badge at the bottom
        answer_parts.append(risk_badge)

        answer_text = "\n\n---\n\n".join(answer_parts)

        # ------------------------------------------------------------------
        # Step 8: Content safety on output (non-blocking)
        # ------------------------------------------------------------------
        output_safety = check_content_safety(answer_text)
        safety_flagged = not output_safety["safe"]
        if safety_flagged:
            print(f"Output safety flag: {output_safety['categories']}")

        # ------------------------------------------------------------------
        # Step 9: Return full Answer
        # ------------------------------------------------------------------
        return Answer(
            answer=answer_text,
            citations=citations,
            structured=structured,
            safety_flagged=safety_flagged,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
