"""
app.py — PolicyIntel Premium  |  Streamlit frontend
"""

import time
import streamlit as st
import requests
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyIntel Premium",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at 15% 15%, #0f1a2e 0%, #000000 55%),
                radial-gradient(ellipse at 85% 85%, #0a1628 0%, #000000 55%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px);
    margin-bottom: 8px;
}

/* ── Stat card ── */
.stat-card {
    background: rgba(56, 189, 248, 0.05);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}
.stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.stat-value { font-size: 1.4rem; font-weight: 700; color: #38bdf8; }
.stat-delta { font-size: 0.72rem; color: #4ade80; margin-top: 2px; }

/* ── Service badge ── */
.svc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.73rem; color: #94a3b8;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 4px 10px; margin: 3px 2px;
}
.svc-dot-on  { width:7px; height:7px; border-radius:50%; background:#4ade80; display:inline-block; }
.svc-dot-off { width:7px; height:7px; border-radius:50%; background:#475569; display:inline-block; }

/* ── Risk pills ── */
.risk-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 18px; border-radius: 999px;
    font-weight: 700; font-size: 0.88rem; letter-spacing: 0.02em;
    margin-bottom: 14px;
}
.risk-low    { background:rgba(34,197,94,0.12);  color:#4ade80; border:1px solid rgba(34,197,94,0.4); }
.risk-medium { background:rgba(234,179,8,0.12);  color:#fbbf24; border:1px solid rgba(234,179,8,0.4); }
.risk-high   { background:rgba(239,68,68,0.12);  color:#f87171; border:1px solid rgba(239,68,68,0.4); }

/* ── Answer section header ── */
.section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin: 18px 0 8px;
}

/* ── Citation card ── */
.cit-card {
    background: linear-gradient(135deg, rgba(56,189,248,0.04) 0%, rgba(99,102,241,0.04) 100%);
    border: 1px solid rgba(56,189,248,0.12);
    border-left: 3px solid #38bdf8;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.cit-source { font-size: 0.78rem; font-weight: 700; color: #38bdf8; }
.cit-body   { font-size: 0.85rem; color: #94a3b8; line-height: 1.55; margin: 8px 0; }
.cit-badge  {
    display: inline-block; font-size: 0.7rem; color: #64748b;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 1px 8px; margin-left: 6px;
}

/* ── KP tag ── */
.kp-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px; padding: 2px 9px;
    font-size: 0.72rem; margin: 2px 3px 2px 0;
}

/* ── Breakdown card ── */
.breakdown-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px;
}
.breakdown-item {
    padding: 6px 0 6px 12px;
    border-left: 2px solid rgba(56,189,248,0.3);
    margin-bottom: 6px; color: #cbd5e1; font-size: 0.88rem;
}

/* ── Mode description ── */
.mode-desc { font-size: 0.75rem; color: #475569; margin-top: 4px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.2); border-radius: 10px; }

/* ── Input box ── */
[data-testid="stChatInput"] > div {
    border-color: rgba(56,189,248,0.25) !important;
    background: rgba(15,23,42,0.8) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_latency"  not in st.session_state: st.session_state.last_latency  = None
if "last_score"    not in st.session_state: st.session_state.last_score    = None
if "total_queries" not in st.session_state: st.session_state.total_queries = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 12px'>
        <div style='font-size:1.5rem;font-weight:800;color:#38bdf8;letter-spacing:-0.02em;'>
            ⚖️ PolicyIntel
        </div>
        <div style='font-size:0.75rem;color:#475569;margin-top:2px;'>
            AI Risk & Compliance Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Live stats from session state
    st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:10px;'>Live Stats</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    lat = st.session_state.last_latency
    score = st.session_state.last_score
    queries = st.session_state.total_queries

    c1.markdown(f"""<div class='stat-card'>
        <div class='stat-label'>Latency</div>
        <div class='stat-value'>{f'{lat:.1f}s' if lat else '—'}</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""<div class='stat-card'>
        <div class='stat-label'>Score</div>
        <div class='stat-value'>{f'{score:.2f}' if score else '—'}</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""<div class='stat-card'>
        <div class='stat-label'>Queries</div>
        <div class='stat-value'>{queries}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Model controls
    st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:10px;'>Model Controls</div>", unsafe_allow_html=True)

    mode = st.select_slider(
        "Reasoning Depth",
        options=["Fast", "Balanced", "Deep"],
        value="Balanced",
        label_visibility="visible",
    )

    MODE_DESC = {
        "Fast":     "3 findings · 5 sources · concise output",
        "Balanced": "5 findings · 10 sources · standard analysis",
        "Deep":     "8 findings · 20 sources · exhaustive review",
    }
    st.markdown(f"<div class='mode-desc'>{MODE_DESC[mode]}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    top_k = st.slider("Context Sources (top_k)", 3, 25, {"Fast": 5, "Balanced": 10, "Deep": 20}[mode])

    st.divider()

    # Azure service status
    st.markdown("<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:8px;'>Azure Services</div>", unsafe_allow_html=True)

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    def _svc_on(env_key): return bool(os.getenv(env_key, "").strip()) and "<" not in os.getenv(env_key, "")

    svcs = [
        ("OpenAI",        _svc_on("AZURE_OPENAI_KEY")),
        ("AI Search",     _svc_on("AZURE_SEARCH_KEY")),
        ("Doc Intel",     _svc_on("AZURE_DOCUMENT_INTELLIGENCE_KEY")),
        ("Language",      _svc_on("AZURE_LANGUAGE_KEY")),
        ("Content Safety",_svc_on("AZURE_CONTENT_SAFETY_KEY")),
        ("App Insights",  _svc_on("APPLICATIONINSIGHTS_CONNECTION_STRING")),
    ]

    badges_html = "".join(
        f"<span class='svc-badge'><span class='{'svc-dot-on' if on else 'svc-dot-off'}'></span>{name}</span>"
        for name, on in svcs
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.divider()

    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.last_latency = None
        st.session_state.last_score = None
        st.session_state.total_queries = 0
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 8px'>
    <div style='font-size:3rem;font-weight:800;letter-spacing:-0.03em;line-height:1.1;'>
        Policy <span style='color:#38bdf8;'>Intelligence</span>
    </div>
    <div style='font-size:1rem;color:#475569;margin-top:10px;font-weight:400;'>
        Ask anything across your insurance and HR policy documents
    </div>
</div>
""", unsafe_allow_html=True)

# ── Policy coverage radar ─────────────────────────────────────────────────────
with st.expander("Policy Coverage Comparison", expanded=False):
    categories = ["Personal Liability", "Life Insurance", "AD&D", "Short-Term Disability", "Personal Property"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[85, 100, 65, 80, 90], theta=categories,
        fill="toself", name="Indexed Policies",
        line=dict(color="#38bdf8"), fillcolor="rgba(56,189,248,0.1)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[60, 55, 50, 65, 70], theta=categories,
        fill="toself", name="Market Average",
        line=dict(color="#6366f1"), fillcolor="rgba(99,102,241,0.08)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#475569")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Chat history replay ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about coverage limits, exclusions, disability, beneficiaries..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_messages = {
            "Fast":     "Fast scan — retrieving top sources...",
            "Balanced": "Searching vector database & semantic index...",
            "Deep":     "Deep analysis — exhaustive document review...",
        }

        with st.status(status_messages[mode], expanded=True) as status:
            t0 = time.time()
            try:
                r = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": prompt, "top_k": top_k, "mode": mode},
                    timeout=120,
                )
                elapsed = time.time() - t0
                status.update(label=f"Analysis complete  ({elapsed:.1f}s)", state="complete", expanded=False)
            except requests.exceptions.ConnectionError:
                status.update(label="Backend unreachable", state="error", expanded=False)
                st.error("Cannot connect to http://127.0.0.1:8000 — start the FastAPI server first.")
                st.stop()
            except requests.exceptions.Timeout:
                status.update(label="Request timed out", state="error", expanded=False)
                st.error("The backend took too long. Try reducing Context Sources or use Fast mode.")
                st.stop()

        # ── Error handling ──
        if r.status_code == 400:
            st.warning(f"Query blocked by Content Safety: {r.json().get('detail', '')}")
            st.stop()
        elif r.status_code != 200:
            st.error(f"Backend error {r.status_code}: {r.text[:400]}")
            st.stop()

        data = r.json()
        structured = data.get("structured") or {}
        citations  = data.get("citations", [])

        # ── Update live stats ──
        st.session_state.last_latency  = elapsed
        st.session_state.total_queries += 1
        top_scores = [c.get("reranker_score") for c in citations if c.get("reranker_score")]
        st.session_state.last_score = max(top_scores) if top_scores else None

        # ── Risk pill ──
        risk_level = structured.get("risk_level", "Medium")
        risk_icon  = {"Low": "●", "Medium": "▲", "High": "■"}.get(risk_level, "●")
        risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(risk_level, "risk-medium")
        st.markdown(f"<span class='risk-pill {risk_class}'>{risk_icon} {risk_level} Risk</span>", unsafe_allow_html=True)

        # ── Content safety warning ──
        if data.get("safety_flagged"):
            st.warning("Content Safety flagged content in the AI response — review carefully.")

        # ── Answer ──
        st.markdown(data.get("answer", "No answer generated."))

        # ── Citations ──
        if citations:
            st.markdown("<div class='section-label'>Source Attribution</div>", unsafe_allow_html=True)
            for cit in citations[:5]:
                reranker  = cit.get("reranker_score")
                page_num  = cit.get("page_number")
                source    = cit.get("source", "Unknown")
                content   = cit.get("content", "")
                kp_list   = (cit.get("key_phrases") or [])[:6]

                score_badge = f"<span class='cit-badge'>score {reranker:.2f}</span>" if reranker else ""
                page_badge  = f"<span class='cit-badge'>p. {page_num}</span>" if page_num else ""
                kp_html     = " ".join(f"<span class='kp-tag'>{kp}</span>" for kp in kp_list)
                preview     = content[:340] + ("…" if len(content) > 340 else "")

                st.markdown(f"""
                <div class='cit-card'>
                    <div class='cit-source'>{source}{score_badge}{page_badge}</div>
                    <div class='cit-body'>{preview}</div>
                    {kp_html}
                </div>
                """, unsafe_allow_html=True)

        # ── Structured Breakdown ──
        if structured:
            with st.expander("Structured Breakdown", expanded=False):
                key_findings = structured.get("key_findings", [])
                coverage_gaps = structured.get("coverage_gaps", [])
                cross_doc = structured.get("cross_document_comparison")

                # Metric row
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level",    risk_level)
                m2.metric("Key Findings",  len(key_findings))
                m3.metric("Coverage Gaps", len(coverage_gaps))

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                col_f, col_g = st.columns(2)
                with col_f:
                    st.markdown("<div class='section-label'>Key Findings</div>", unsafe_allow_html=True)
                    for item in key_findings:
                        st.markdown(f"<div class='breakdown-item'>{item}</div>", unsafe_allow_html=True)
                    if not key_findings:
                        st.caption("None identified.")

                with col_g:
                    st.markdown("<div class='section-label'>Coverage Gaps</div>", unsafe_allow_html=True)
                    for item in coverage_gaps:
                        st.markdown(f"<div class='breakdown-item'>{item}</div>", unsafe_allow_html=True)
                    if not coverage_gaps:
                        st.caption("No gaps identified.")

                if cross_doc:
                    st.markdown("<div class='section-label' style='margin-top:16px'>Cross-Document Comparison</div>", unsafe_allow_html=True)
                    st.info(cross_doc)

        # ── Save to history ──
        st.session_state.messages.append({"role": "assistant", "content": data.get("answer", "")})
