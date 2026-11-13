"""
GrantBridge – AI Grant & Funding Finder for Startups
Main Streamlit Application Entry Point
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="GrantBridge – AI Grant Finder",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "GrantBridge – AI-Powered Grant & Funding Finder powered by IBM Granite",
    },
)

import time
import json
import logging
import uuid
from typing import Dict, Any, Optional, List

# GrantBridge modules
from core.config import config
from core.granite_client import GraniteClient
from core.knowledge_base import GrantKnowledgeBase
from guardrails.guardrail_engine import GuardrailEngine
from agents.multi_agent_pipeline import run_grant_search, WorkflowState, AgentStatus
from evaluation.eval_engine import EvaluationEngine

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def init_core_services():
    """Initialise heavy services once per process."""
    granite = GraniteClient(config)
    kb = GrantKnowledgeBase(config)
    guardrails = GuardrailEngine(config)
    eval_engine = EvaluationEngine()
    return granite, kb, guardrails, eval_engine


def init_session():
    defaults = {
        "workflow_history": [],
        "current_state": None,
        "agents": {},
        "eval_results": {},
        "search_count": 0,
        "total_start_time": time.time(),
        "user_id": str(uuid.uuid4())[:8],
        "human_approval_pending": False,
        "pending_state": None,
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ---- Brand colours ---- */
    :root {
        --gb-blue: #1A6CE8;
        --gb-dark: #0D1B2A;
        --gb-green: #22C55E;
        --gb-amber: #F59E0B;
        --gb-red: #EF4444;
        --gb-surface: #F8FAFC;
        --gb-border: #E2E8F0;
    }
    /* Header */
    .gb-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A3A5C 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .gb-header h1 { margin: 0; font-size: 2rem; }
    .gb-header p  { margin: 0.25rem 0 0; opacity: 0.8; font-size: 0.95rem; }
    /* KPI cards */
    .kpi-card {
        background: white;
        border: 1px solid var(--gb-border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: var(--gb-blue); }
    .kpi-label { font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    /* Grant cards */
    .grant-card {
        background: white;
        border: 1px solid var(--gb-border);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .grant-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .grant-name { font-size: 1.1rem; font-weight: 600; color: #1E293B; }
    .grant-amount { font-size: 1rem; color: var(--gb-green); font-weight: 600; }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.3rem;
    }
    .badge-green { background: #D1FAE5; color: #065F46; }
    .badge-blue  { background: #DBEAFE; color: #1E40AF; }
    .badge-amber { background: #FEF3C7; color: #92400E; }
    .badge-red   { background: #FEE2E2; color: #991B1B; }
    /* Guardrail status */
    .guardrail-pass { color: var(--gb-green); font-weight: 600; }
    .guardrail-warn { color: var(--gb-amber); font-weight: 600; }
    .guardrail-fail { color: var(--gb-red);   font-weight: 600; }
    /* Agent timeline */
    .agent-step {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.5rem 0; border-bottom: 1px solid var(--gb-border);
    }
    .agent-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: var(--gb-green); flex-shrink: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def render_sidebar(granite: GraniteClient):
    with st.sidebar:
        st.markdown("## 🌉 GrantBridge")
        st.caption("AI Grant & Funding Finder")
        st.divider()

        mode = "🟡 Simulation" if granite.is_simulation else "🟢 IBM Granite (Live)"
        st.info(f"**LLM Mode:** {mode}", icon="🤖")
        st.caption(f"Model: `{config.GRANITE_MODEL_ID}`")

        st.divider()
        st.markdown("### 📂 Navigation")

        page = st.radio(
            "Select Page",
            options=[
                "🔍 Grant Finder",
                "📊 LLM Evaluation Dashboard",
                "🛡️ Guardrails Monitor",
                "🤖 Agent Pipeline",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        with st.expander("⚙️ Session Info"):
            st.text(f"User ID: {st.session_state.user_id}")
            st.text(f"Searches: {st.session_state.search_count}")
            st.text(f"Requests: {granite.usage.total_requests}")
            cost = granite.usage.total_cost_usd
            st.text(f"Est. Cost: ${cost:.4f}")
        return page


# ─────────────────────────────────────────────────────────────
# Page: Grant Finder
# ─────────────────────────────────────────────────────────────

def render_grant_finder(granite, kb, guardrails, eval_engine):
    st.markdown("""
    <div class="gb-header">
        <h1>🌉 GrantBridge</h1>
        <p>AI-Powered Grant & Funding Finder · Powered by IBM Granite · Secured by 11 Guardrail Categories</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Startup Profile Form ────────────────────────────────
    with st.form("startup_profile_form"):
        st.subheader("📋 Startup Profile")
        c1, c2 = st.columns(2)
        with c1:
            startup_name = st.text_input("Startup Name *", placeholder="e.g. MedAI Technologies")
            industry = st.selectbox("Industry / Sector *", [
                "AI & Machine Learning", "Healthtech / Biotech", "Fintech",
                "Edtech", "Agritech", "Cleantech / Greentech", "Deep Tech",
                "IoT / Hardware", "SaaS / Enterprise Tech", "E-commerce / D2C", "Other"
            ])
            location = st.selectbox("Location *", [
                "India", "United States", "European Union", "United Kingdom",
                "Singapore", "Canada", "Australia", "Other"
            ])
        with c2:
            stage = st.selectbox("Startup Stage *", ["Idea", "MVP", "Early-stage", "Growth"])
            funding_req = st.number_input(
                "Funding Requirement (USD) *", min_value=1000, max_value=10_000_000,
                value=50000, step=5000, format="%d"
            )
            founding_year = st.number_input(
                "Founding Year", min_value=2010, max_value=2025, value=2023
            )

        st.subheader("📄 Additional Details")
        c3, c4 = st.columns(2)
        with c3:
            description = st.text_area(
                "Brief Description",
                placeholder="Describe your startup's product, problem solved, and target market…",
                height=90,
            )
            dpiit = st.checkbox("DPIIT Recognised (India)")
        with c4:
            user_query = st.text_area(
                "What are you looking for? *",
                placeholder="e.g. Looking for seed funding to build an MVP for AI-powered diagnostics…",
                height=90,
            )
            uploaded_file = st.file_uploader("Upload Business Plan (optional)", type=["pdf", "docx", "txt"])

        submitted = st.form_submit_button("🔍 Find Matching Grants", use_container_width=True, type="primary")

    if not submitted:
        _render_sample_grants(kb)
        return

    # ── Input Validation ────────────────────────────────────
    if not startup_name or not user_query:
        st.error("Please fill in all required fields (*).")
        return

    startup_data = {
        "startup_name": startup_name,
        "industry": industry,
        "location": location,
        "stage": stage,
        "funding_requirement": funding_req,
        "founding_year": founding_year,
        "description": description,
        "dpiit_recognised": dpiit,
    }

    # File validation
    file_name = uploaded_file.name if uploaded_file else None
    file_size = len(uploaded_file.getvalue()) if uploaded_file else None

    with st.spinner("🛡️ Running input guardrails…"):
        input_check = guardrails.run_input_checks(
            user_query, startup_data, file_name, file_size
        )

    if input_check.blocked:
        st.error(f"⛔ Input Blocked: {input_check.message}")
        for v in input_check.violations:
            st.warning(f"• {v}")
        return

    if input_check.violations:
        with st.expander("⚠️ Input Warnings (non-blocking)", expanded=False):
            for v in input_check.violations:
                st.warning(f"• {v}")

    # ── Run Agent Pipeline ───────────────────────────────────
    progress = st.progress(0, text="Initialising agents…")
    start_time = time.time()

    with st.status("🤖 GrantBridge Agents Working…", expanded=True) as status_box:
        st.write("🔎 **Grant Discovery Agent** – Searching knowledge base…")
        progress.progress(20, text="Discovery Agent…")
        time.sleep(0.3)

        st.write("✅ **Eligibility Agent** – Checking eligibility criteria…")
        progress.progress(40, text="Eligibility Agent…")
        time.sleep(0.2)

        st.write("📚 **RAG Agent** – Retrieving verified context…")
        progress.progress(60, text="RAG Agent…")
        time.sleep(0.2)

        st.write("💡 **Recommendation Agent** – Generating personalised recommendations…")
        progress.progress(80, text="Recommendation Agent…")

        try:
            state, agents = run_grant_search(
                startup_profile=startup_data,
                user_query=user_query,
                granite_client=granite,
                guardrail_engine=guardrails,
                knowledge_base=kb,
            )
            progress.progress(100, text="Complete!")
            status_box.update(label="✅ Agent Pipeline Complete", state="complete")
        except Exception as exc:
            status_box.update(label="❌ Pipeline Error", state="error")
            st.error(f"Pipeline error: {exc}")
            logger.error(f"Pipeline exception: {exc}", exc_info=True)
            return

    elapsed = time.time() - start_time
    st.session_state.current_state = state
    st.session_state.agents = agents
    st.session_state.search_count += 1
    st.session_state.workflow_history.append({
        "query": user_query,
        "startup": startup_name,
        "timestamp": time.strftime("%H:%M:%S"),
        "grants_found": len(state.recommendations),
        "elapsed_s": round(elapsed, 2),
    })

    # ── Human-in-the-Loop Approval ──────────────────────────
    if state.requires_human_approval:
        st.session_state.human_approval_pending = True
        st.session_state.pending_state = state
        st.warning(
            "⏳ **Human Approval Required** – GrantBridge has prepared grant recommendations. "
            "Please review and approve before proceeding."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve Recommendations", type="primary"):
                state.human_approved = True
                state.status = AgentStatus.COMPLETED
                st.session_state.human_approval_pending = False
                st.rerun()
        with col_b:
            if st.button("❌ Reject / Revise"):
                st.session_state.human_approval_pending = False
                st.info("Please refine your query and search again.")
                return

    # ── Results Display ──────────────────────────────────────
    st.divider()
    _render_results(state, elapsed)

    # Run evaluation in background
    with st.spinner("Computing evaluation metrics…"):
        eval_results = eval_engine.run_full_evaluation(
            granite_client=granite,
            guardrail_engine=guardrails,
            agents=agents,
            workflow_state=state,
        )
        st.session_state.eval_results = eval_results


def _render_sample_grants(kb):
    st.divider()
    st.subheader("📌 Available Grants in Knowledge Base")
    grants = kb.get_all_grants()
    for g in grants[:6]:
        with st.expander(f"🏆 {g.get('name', 'Unknown')} — {g.get('currency', 'USD')} {g.get('amount_max', 0):,}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Description:** {g.get('description', '')}")
                st.markdown(f"**Eligibility:** {g.get('eligibility', '')}")
                st.markdown(f"**Documents Required:** {', '.join(g.get('documents', []))}")
            with c2:
                st.markdown(f"**Deadline:** `{g.get('deadline', 'N/A')}`")
                st.markdown(f"**Stage:** {', '.join(g.get('stage', []))}")
                st.markdown(f"**Sectors:** {', '.join(g.get('sectors', []))}")
                if g.get("source"):
                    st.markdown(f"[🔗 Official Portal]({g['source']})")


def _render_results(state: WorkflowState, elapsed: float):
    recommendations = state.recommendations
    total_found = len(state.discovered_grants)
    total_eligible = len(state.eligible_grants)
    total_rec = len(recommendations)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, total_found, "Grants Found"),
        (k2, total_eligible, "Eligible Grants"),
        (k3, total_rec, "Recommendations"),
        (k4, f"{elapsed:.1f}s", "Search Time"),
        (k5, state.task_id, "Task ID"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Agent recommendation text
    if state.final_response:
        with st.expander("💬 GrantBridge AI Analysis", expanded=True):
            st.markdown(state.final_response)
            st.caption("⚠️ Verify all grant details on official portals. Amounts and deadlines are approximate.")

    # Grant recommendation cards
    if recommendations:
        st.subheader(f"🏆 Top {len(recommendations)} Grant Recommendations")
        for idx, grant in enumerate(recommendations, 1):
            elig_class = "badge-green" if grant.get("eligibility_status") == "eligible" else "badge-amber"
            elig_label = grant.get("eligibility_status", "pending").title()
            verified_badge = '<span class="badge badge-blue">✓ Verified</span>' if grant.get("verified") else ""
            amount_str = (
                f"{grant.get('currency', 'USD')} {grant.get('amount_min', 0):,} – "
                f"{grant.get('amount_max', 0):,}"
            )
            relevance_pct = int(grant.get("relevance_score", 0.75) * 100)

            st.markdown(f"""
            <div class="grant-card">
                <div style="display:flex;justify-content:space-between;align-items:start">
                    <div>
                        <span class="grant-name">#{idx} {grant.get('name', 'Unknown Grant')}</span>
                        {verified_badge}
                        <span class="badge {elig_class}">{elig_label}</span>
                    </div>
                    <div class="grant-amount">{amount_str}</div>
                </div>
                <div style="margin-top:0.5rem;color:#475569;font-size:0.88rem">
                    📅 Deadline: <b>{grant.get('deadline', 'N/A')}</b> &nbsp;|&nbsp;
                    🎯 Relevance: <b>{relevance_pct}%</b> &nbsp;|&nbsp;
                    🏭 Sectors: {', '.join(grant.get('sectors', [])[:3])}
                </div>
                <div style="margin-top:0.4rem;font-size:0.85rem;color:#64748B">
                    {grant.get('match_reason', '')}
                </div>
                {"<a href='" + grant['source'] + "' target='_blank' style='font-size:0.82rem;color:#1A6CE8'>🔗 Official Portal</a>" if grant.get('source') else ""}
            </div>
            """, unsafe_allow_html=True)

    # Errors
    if state.errors:
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for err in state.errors:
                st.warning(err)


# ─────────────────────────────────────────────────────────────
# Page: LLM Evaluation Dashboard
# ─────────────────────────────────────────────────────────────

def render_evaluation_dashboard(granite, guardrails, eval_engine):
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.markdown("""
    <div class="gb-header">
        <h1>📊 LLM Evaluation & Agent Performance Dashboard</h1>
        <p>Comprehensive evaluation across all 6 metric categories · Real-time monitoring · IBM Granite</p>
    </div>
    """, unsafe_allow_html=True)

    eval_results = st.session_state.get("eval_results", {})
    agents = st.session_state.get("agents", {})
    state = st.session_state.get("current_state", None)

    # Run evaluation if not yet done
    if not eval_results:
        if not agents:
            st.info("💡 Run a grant search first to generate real evaluation metrics. Showing baseline demo data.")
            # Build demo agents
            from agents.multi_agent_pipeline import build_pipeline
            demo_granite = GraniteClient(config)
            demo_kb = GrantKnowledgeBase(config)
            demo_guardrails = GuardrailEngine(config)
            _, agents = build_pipeline(demo_granite, demo_guardrails, demo_kb)
            granite_for_eval = demo_granite
            guardrails_for_eval = demo_guardrails
        else:
            granite_for_eval = granite
            guardrails_for_eval = guardrails

        with st.spinner("Computing evaluation metrics…"):
            eval_results = eval_engine.run_full_evaluation(
                granite_client=granite_for_eval,
                guardrail_engine=guardrails_for_eval,
                agents=agents,
                workflow_state=state,
            )
            st.session_state.eval_results = eval_results

    overall_score = eval_engine.get_overall_score()

    # ── Overall KPIs ─────────────────────────────────────────
    st.subheader("🎯 Overall System Performance")

    # Mode banner
    sim_mode = granite.is_simulation
    if sim_mode:
        st.info(
            "ℹ️ **Simulation Mode** – IBM Granite credentials not configured. "
            "Metrics are computed from actual component logic with simulated LLM responses. "
            "Connect real IBM WatsonX credentials in `.env` for live values.",
            icon="🧪"
        )
    else:
        st.success("✅ **Live Mode** – Connected to IBM WatsonX Granite. All metrics reflect real API calls.", icon="🔗")

    score_color = "#22C55E" if overall_score >= 0.75 else "#F59E0B" if overall_score >= 0.5 else "#EF4444"
    st.markdown(
        f'<div style="text-align:center;padding:1.5rem;background:white;border-radius:12px;border:2px solid {score_color};">'
        f'<div style="font-size:3rem;font-weight:700;color:{score_color}">{overall_score:.1%}</div>'
        f'<div style="font-size:1rem;color:#64748B">Overall System Score</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category Tabs ─────────────────────────────────────────
    tabs = st.tabs([
        "1️⃣ Accuracy & Quality",
        "2️⃣ RAG Metrics",
        "3️⃣ Agentic AI",
        "4️⃣ Safety & Guardrails",
        "5️⃣ Performance & Cost",
        "6️⃣ Multi-Agent System",
        "📈 Summary Charts",
    ])

    # ── Tab 1: Accuracy & Quality ─────────────────────────────
    with tabs[0]:
        _render_category_tab(
            eval_results.get("accuracy"),
            "Accuracy and Quality Metrics",
            {
                "accuracy": "Overall correctness of generated answers",
                "precision": "How many generated relevant results are actually correct",
                "recall": "How many relevant results are successfully retrieved",
                "f1_score": "Harmonic mean of precision and recall",
                "exact_match": "Exact match between generated and expected answer",
                "semantic_similarity": "Similarity between generated and reference answers",
                "rouge_l": "ROUGE-L overlap with reference text",
                "bleu": "BLEU n-gram precision against reference text",
            },
        )

    # ── Tab 2: RAG ────────────────────────────────────────────
    with tabs[1]:
        _render_category_tab(
            eval_results.get("rag"),
            "RAG Evaluation Metrics",
            {
                "context_precision": "Whether retrieved documents are relevant",
                "context_recall": "Whether relevant information is retrieved",
                "faithfulness": "Answer is supported by retrieved context",
                "answer_relevancy": "Relevance of answer to user query",
                "context_relevancy": "Relevance of retrieved context",
                "groundedness": "Generated responses grounded in source data",
            },
        )

    # ── Tab 3: Agentic AI ─────────────────────────────────────
    with tabs[2]:
        result = eval_results.get("agent")
        if result:
            st.subheader("Agentic AI Evaluation Metrics")
            _render_metrics_grid(result.metrics)

            individual = result.details.get("individual_agents", {})
            if individual:
                st.subheader("Individual Agent Performance")
                agent_df = pd.DataFrame([
                    {
                        "Agent": aid.replace("_", " ").title(),
                        "Completion Rate": f"{v['task_completion_rate']:.1%}",
                        "Tool Accuracy": f"{v['tool_call_accuracy']:.1%}",
                        "Tool Calls": v["tool_calls"],
                        "Latency (ms)": f"{v['latency_ms']:.0f}",
                        "Iterations": v["iterations"],
                        "Tasks Done": v["tasks_completed"],
                        "Tasks Failed": v["tasks_failed"],
                        "Reasoning Quality": f"{v['reasoning_quality']:.1%}",
                    }
                    for aid, v in individual.items()
                ])
                st.dataframe(agent_df, use_container_width=True)

                # Radar chart for agents
                fig = _build_agent_radar(individual)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Safety & Guardrails ────────────────────────────
    with tabs[3]:
        result = eval_results.get("safety")
        if result:
            st.subheader("Safety and Guardrail Metrics")
            _render_metrics_grid(result.metrics, invert_bad=True)

            details = result.details
            st.subheader("Guardrail Check Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Checks", details.get("total_guardrail_checks", 0))
            with c2:
                st.metric("✅ Passed", details.get("passed", 0), delta_color="normal")
            with c3:
                st.metric("⚠️ Warnings", details.get("warnings", 0))
            with c4:
                st.metric("🚫 Blocked", details.get("blocked", 0))

            by_cat = details.get("by_category", {})
            if by_cat:
                cat_df = pd.DataFrame([
                    {"Category": cat, "Passed": v.get("passed", 0),
                     "Warnings": v.get("warnings", 0), "Blocked": v.get("blocked", 0)}
                    for cat, v in by_cat.items()
                ])
                fig = px.bar(
                    cat_df, x="Category", y=["Passed", "Warnings", "Blocked"],
                    barmode="stack", color_discrete_map={"Passed": "#22C55E", "Warnings": "#F59E0B", "Blocked": "#EF4444"},
                    title="Guardrail Results by Category",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Performance & Cost ─────────────────────────────
    with tabs[4]:
        result = eval_results.get("performance")
        if result:
            st.subheader("Performance and Cost Metrics")
            m = result.metrics

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.metric("Avg Response Latency", f"{m.get('response_latency_ms', 0):.0f} ms")
                st.metric("TTFT", f"{m.get('time_to_first_token_ms', 0):.0f} ms")
                st.metric("Tokens/Second", f"{m.get('tokens_per_second', 0):.1f}")
            with pc2:
                st.metric("Input Tokens (total)", f"{int(m.get('input_tokens_total', 0)):,}")
                st.metric("Output Tokens (total)", f"{int(m.get('output_tokens_total', 0)):,}")
                st.metric("Error Rate", f"{m.get('error_rate', 0):.1%}")
            with pc3:
                st.metric("Cost / Request", f"${m.get('cost_per_request_usd', 0):.5f}")
                st.metric("Total Cost", f"${m.get('total_cost_usd', 0):.4f}")
                st.metric("Throughput", f"{m.get('throughput_req_per_min', 0):.1f} req/min")

            # Latency gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=m.get("response_latency_ms", 500),
                delta={"reference": 2000},
                title={"text": "Response Latency (ms)"},
                gauge={
                    "axis": {"range": [0, 5000]},
                    "bar": {"color": "#1A6CE8"},
                    "steps": [
                        {"range": [0, 500], "color": "#D1FAE5"},
                        {"range": [500, 2000], "color": "#FEF3C7"},
                        {"range": [2000, 5000], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": 2000},
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

            details = result.details
            st.caption(
                f"Model: `{details.get('model_id', 'N/A')}` · "
                f"Requests: {details.get('total_requests', 0)} · "
                f"Simulation: {'Yes' if details.get('simulation_mode') else 'No'}"
            )

    # ── Tab 6: Multi-Agent System ─────────────────────────────
    with tabs[5]:
        result = eval_results.get("multi_agent")
        if result:
            st.subheader("Multi-Agent System Evaluation")
            _render_metrics_grid(result.metrics)

            ind_acc = result.details.get("individual_agent_accuracy", {})
            if ind_acc:
                st.subheader("Individual Agent Accuracy")
                cols = st.columns(len(ind_acc))
                for i, (aid, acc) in enumerate(ind_acc.items()):
                    with cols[i]:
                        color = "#22C55E" if acc >= 0.8 else "#F59E0B" if acc >= 0.6 else "#EF4444"
                        st.markdown(
                            f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{acc:.1%}</div>'
                            f'<div class="kpi-label">{aid.replace("_", " ").title()}</div></div>',
                            unsafe_allow_html=True,
                        )

                fig = px.bar(
                    x=list(ind_acc.keys()),
                    y=list(ind_acc.values()),
                    labels={"x": "Agent", "y": "Accuracy"},
                    title="Individual Agent Accuracy",
                    color=list(ind_acc.values()),
                    color_continuous_scale=["#EF4444", "#F59E0B", "#22C55E"],
                    range_color=[0, 1],
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            st.metric("Total Execution Time", f"{result.details.get('total_elapsed_ms', 0):.0f} ms")
            st.metric("Total Agents", result.details.get("total_agents", 0))

    # ── Tab 7: Summary Charts ─────────────────────────────────
    with tabs[6]:
        st.subheader("📈 Evaluation Summary")
        _render_summary_charts(eval_results)


def _render_category_tab(result, title: str, metric_descriptions: Dict[str, str]):
    if not result:
        st.info("Run a grant search to populate evaluation metrics.")
        return
    st.subheader(title)
    _render_metrics_grid(result.metrics)

    # Description table
    with st.expander("📖 Metric Descriptions", expanded=False):
        import pandas as pd
        rows = [
            {
                "Metric": k.replace("_", " ").title(),
                "Value": f"{result.metrics.get(k, 0):.4f}",
                "Description": v,
            }
            for k, v in metric_descriptions.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_metrics_grid(metrics: Dict[str, float], invert_bad: bool = False):
    SKIP_KEYS = {"input_tokens_total", "output_tokens_total", "total_execution_time_ms"}
    items = [(k, v) for k, v in metrics.items() if k not in SKIP_KEYS and isinstance(v, (int, float))]
    per_row = 4
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for j, (k, v) in enumerate(items[i:i + per_row]):
            with cols[j]:
                display_v = v
                is_pct = 0.0 <= v <= 1.0 and not any(
                    x in k for x in ["latency", "time", "token", "cost", "throughput", "calls", "count"]
                )
                if is_pct:
                    is_bad = any(x in k for x in ["error", "hallucination", "toxicity", "bias", "pii", "violation"])
                    color = "#EF4444" if (is_bad and v > 0.1) else "#22C55E" if (not is_bad and v >= 0.7) else "#F59E0B"
                    pct_str = f"{v:.1%}"
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{pct_str}</div>'
                        f'<div class="kpi-label">{k.replace("_", " ").title()}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric(k.replace("_", " ").title(), f"{v:,.2f}" if isinstance(v, float) else v)


def _build_agent_radar(individual: Dict):
    import plotly.graph_objects as go

    categories = ["Completion Rate", "Tool Accuracy", "Reasoning Quality"]
    fig = go.Figure()
    for aid, v in individual.items():
        fig.add_trace(go.Scatterpolar(
            r=[v["task_completion_rate"], v["tool_call_accuracy"], v["reasoning_quality"]],
            theta=categories,
            fill="toself",
            name=aid.replace("_", " ").title(),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Agent Performance Radar",
    )
    return fig


def _render_summary_charts(eval_results: Dict):
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    if not eval_results:
        st.info("No evaluation data yet. Run a grant search first.")
        return

    # Radar chart: one spoke per evaluation category
    categories, scores = [], []
    for key, result in eval_results.items():
        if result:
            vals = [v for k, v in result.metrics.items()
                    if isinstance(v, float) and 0 <= v <= 1
                    and not any(x in k for x in ["error", "hallucination", "toxicity", "bias", "pii"])]
            if vals:
                categories.append(result.category.split(" ")[0])
                scores.append(round(sum(vals) / len(vals), 4))

    if categories:
        fig = go.Figure(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#1A6CE8",
            fillcolor="rgba(26,108,232,0.15)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Evaluation Category Summary (avg score per category)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart: all metrics at a glance
    all_metrics = []
    for result in eval_results.values():
        if result:
            for k, v in result.metrics.items():
                if isinstance(v, float) and 0 <= v <= 1:
                    all_metrics.append({
                        "Category": result.category[:20],
                        "Metric": k.replace("_", " ").title()[:22],
                        "Score": round(v, 4),
                    })

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        fig2 = px.bar(
            df, x="Metric", y="Score", color="Category",
            title="All Evaluation Metrics at a Glance",
            height=450,
        )
        fig2.update_xaxes(tickangle=45)
        fig2.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig2, use_container_width=True)

    # Metric table
    st.subheader("📋 Full Metrics Table")
    all_rows = []
    for result in eval_results.values():
        if result:
            for k, v in result.metrics.items():
                display = f"{v:.1%}" if isinstance(v, float) and 0 <= v <= 1 else f"{v:,.2f}" if isinstance(v, float) else str(v)
                status = "🟢" if isinstance(v, float) and v >= 0.7 and 0 <= v <= 1 else "🟡" if isinstance(v, float) and v >= 0.4 and v <= 1 else "⚪"
                all_rows.append({
                    "Category": result.category,
                    "Metric": k.replace("_", " ").title(),
                    "Value": display,
                    "Status": status,
                    "Data Source": "Measured" if not GraniteClient(config).is_simulation else "Simulated",
                })
    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Page: Guardrails Monitor
# ─────────────────────────────────────────────────────────────

def render_guardrails_monitor(guardrails):
    st.markdown("""
    <div class="gb-header">
        <h1>🛡️ Guardrails Monitor</h1>
        <p>Real-time monitoring of all 11 guardrail categories</p>
    </div>
    """, unsafe_allow_html=True)

    stats = guardrails.stats
    total = max(1, stats["total_checks"])

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Checks", stats["total_checks"])
    with c2: st.metric("✅ Passed", stats["passed"])
    with c3: st.metric("⚠️ Warnings", stats["warnings"])
    with c4: st.metric("🚫 Blocked", stats["blocked"])

    st.divider()

    guardrail_categories = [
        ("1. Input Guardrails", "Validates user inputs, PII detection, injection prevention"),
        ("2. Output Guardrails", "Validates AI responses, hallucination detection"),
        ("3. Eligibility Guardrails", "Verifies startup-grant eligibility matching"),
        ("4. RAG Guardrails", "Validates retrieved document relevance and freshness"),
        ("5. Tool Execution Guardrails", "Controls agent tool access and rate limiting"),
        ("6. Data Guardrails", "Protects startup data and prevents unauthorised access"),
        ("7. Security Guardrails", "Jailbreak detection, SQL injection prevention"),
        ("8. Operational Guardrails", "Execution limits, loop detection, timeouts"),
        ("9. Human-in-the-Loop", "Approval workflows for critical actions"),
        ("10. Multi-Agent Guardrails", "Agent coordination and handoff validation"),
        ("11. Ethical & Compliance", "Bias detection, fairness, audit trails"),
    ]

    st.subheader("Guardrail Categories Status")
    by_cat = stats.get("by_category", {})
    for cat_name, desc in guardrail_categories:
        cat_key = cat_name.split(". ")[1].split(" ")[0]
        cat_stats = by_cat.get(cat_key, {"passed": 0, "warnings": 0, "blocked": 0})
        total_cat = sum(cat_stats.values()) or 1
        pass_rate = cat_stats.get("passed", 0) / total_cat
        status_icon = "🟢" if pass_rate >= 0.9 else "🟡" if pass_rate >= 0.5 else "🔴"

        with st.expander(f"{status_icon} {cat_name}"):
            st.caption(desc)
            cs1, cs2, cs3 = st.columns(3)
            with cs1: st.metric("Passed", cat_stats.get("passed", 0))
            with cs2: st.metric("Warnings", cat_stats.get("warnings", 0))
            with cs3: st.metric("Blocked", cat_stats.get("blocked", 0))

    # Interactive guardrail test
    st.divider()
    st.subheader("🧪 Test Input Guardrails")
    with st.form("guardrail_test"):
        test_input = st.text_area("Test Input", placeholder="Enter text to test against guardrails…")
        test_startup = st.checkbox("Include startup data validation")
        if test_startup:
            tc1, tc2 = st.columns(2)
            with tc1:
                ts_name = st.text_input("Test Startup Name", value="TestCo")
                ts_industry = st.text_input("Test Industry", value="AI")
            with tc2:
                ts_location = st.text_input("Test Location", value="India")
                ts_funding = st.text_input("Test Funding", value="50000")
        run_test = st.form_submit_button("Run Guardrail Test")

    if run_test and test_input:
        sd = None
        if test_startup:
            sd = {
                "startup_name": ts_name,
                "industry": ts_industry,
                "location": ts_location,
                "funding_requirement": ts_funding,
            }
        result = guardrails.run_input_checks(test_input, sd)
        if result.blocked:
            st.error(f"🚫 BLOCKED: {result.message}")
        elif result.violations:
            st.warning(f"⚠️ Warnings: {', '.join(result.violations)}")
        else:
            st.success(f"✅ Passed: {result.message}")

        with st.expander("Test Details"):
            st.json({
                "passed": result.passed,
                "blocked": result.blocked,
                "category": result.category,
                "message": result.message,
                "violations": result.violations,
            })


# ─────────────────────────────────────────────────────────────
# Page: Agent Pipeline
# ─────────────────────────────────────────────────────────────

def render_agent_pipeline():
    st.markdown("""
    <div class="gb-header">
        <h1>🤖 Agent Pipeline Monitor</h1>
        <p>Multi-agent workflow execution trace · LangGraph-style orchestration</p>
    </div>
    """, unsafe_allow_html=True)

    state = st.session_state.get("current_state")
    agents = st.session_state.get("agents", {})
    history = st.session_state.get("workflow_history", [])

    if not state:
        st.info("💡 Run a grant search to see the agent pipeline execution trace.")
        st.subheader("Agent Architecture")
        st.markdown("""
        ```
        User Input
             │
        ┌────▼──────────────────────────────────────────┐
        │          Supervisor Agent (LangGraph)          │
        │   Routes tasks · Monitors · Approves outputs  │
        └────┬──────────────────────────────────────────┘
             │
        ┌────▼──────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐
        │  Grant    │→ │  Eligibility │→ │   RAG    │→ │  Recommendation  │
        │ Discovery │  │    Agent     │  │  Agent   │  │      Agent       │
        └───────────┘  └──────────────┘  └──────────┘  └──────────────────┘
             │
        ┌────▼──────────────────────────────────────────┐
        │        Output Guardrails + Human Approval     │
        └───────────────────────────────────────────────┘
        ```
        """)
        return

    # Pipeline status
    st.subheader(f"Task: `{state.task_id}`")
    st.metric("Status", state.status.value.upper())

    # Agent execution timeline
    st.subheader("🔄 Agent Execution Timeline")
    agent_order = ["supervisor", "grant_discovery", "eligibility", "rag", "recommendation"]
    for i, agent_id in enumerate(agent_order):
        agent = agents.get(agent_id)
        if not agent:
            continue
        m = agent.metrics
        completed = m.tasks_completed > 0
        icon = "✅" if completed else "⏳"
        with st.expander(f"{icon} {agent_id.replace('_', ' ').title()} — {m.latency_ms:.0f} ms"):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1: st.metric("Tasks Completed", m.tasks_completed)
            with ac2: st.metric("Tasks Failed", m.tasks_failed)
            with ac3: st.metric("Tool Calls", m.tool_calls)
            with ac4: st.metric("Latency (ms)", f"{m.latency_ms:.0f}")

            if m.reasoning_steps:
                st.markdown("**Reasoning Steps:**")
                for step in m.reasoning_steps:
                    st.markdown(f'<div class="agent-step"><div class="agent-dot"></div><code>{step}</code></div>',
                                unsafe_allow_html=True)

    # Agent outputs
    if state.agent_outputs:
        st.subheader("📦 Agent Outputs")
        for agent_id, output in state.agent_outputs.items():
            with st.expander(f"Output: {agent_id.replace('_', ' ').title()}"):
                st.json({k: v for k, v in output.items() if k != "llm_analysis" and k != "llm_summary"})
                if "llm_analysis" in output or "llm_summary" in output:
                    llm_txt = output.get("llm_analysis") or output.get("llm_summary", "")
                    if llm_txt:
                        st.markdown("**LLM Output:**")
                        st.markdown(llm_txt)

    # Search history
    if history:
        st.subheader("📋 Search History")
        import pandas as pd
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────

def render_about():
    st.markdown("""
    <div class="gb-header">
        <h1>ℹ️ About GrantBridge</h1>
        <p>Problem Statement #18 · IBM SkillsBuild · AI Grant & Funding Finder for Startups</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🌉 GrantBridge

    **GrantBridge** is an AI-powered Grant and Funding Finder for startups, built for IBM SkillsBuild
    (Problem Statement #18). It uses **IBM Granite** via WatsonX AI as its core LLM, with a
    multi-agent architecture powered by **LangGraph**, **ChromaDB** for RAG, and **Streamlit** for the UI.

    ---

    ### 🏗️ Architecture

    | Component | Technology |
    |-----------|------------|
    | LLM | IBM Granite (WatsonX AI) |
    | Agent Orchestration | LangGraph / LangChain |
    | Vector Database | ChromaDB |
    | Web Scraping | BeautifulSoup4 |
    | Document Processing | PyMuPDF |
    | Frontend | Streamlit |
    | Cloud Platform | IBM Cloud Lite |

    ---

    ### 🤖 Agent System

    1. **Supervisor Agent** – Routes tasks, monitors coordination, approves outputs
    2. **Grant Discovery Agent** – Finds funding opportunities from KB and web
    3. **Eligibility Agent** – Verifies startup-grant eligibility criteria
    4. **RAG Agent** – Retrieves and validates contextual grant information
    5. **Recommendation Agent** – Generates ranked, personalised recommendations

    ---

    ### 🛡️ 11 Guardrail Categories

    1. Input Guardrails · 2. Output Guardrails · 3. Eligibility Guardrails ·
    4. RAG & Knowledge Base Guardrails · 5. Tool Execution Guardrails ·
    6. Data Guardrails · 7. Security Guardrails · 8. Operational Guardrails ·
    9. Human-in-the-Loop Guardrails · 10. Multi-Agent Guardrails ·
    11. Ethical & Compliance Guardrails

    ---

    ### 📊 LLM Evaluation (6 Categories)

    1. Accuracy & Quality Metrics (ROUGE, BLEU, Semantic Similarity, F1, etc.)
    2. RAG Evaluation Metrics (Faithfulness, Groundedness, Context Precision, etc.)
    3. Agentic AI Metrics (Task Completion, Tool Accuracy, Loop Detection, etc.)
    4. Safety & Guardrail Metrics (Hallucination Rate, PII Leakage, Toxicity, etc.)
    5. Performance & Cost Metrics (Latency, TTFT, Throughput, Cost per Request)
    6. Multi-Agent System Metrics (E2E Completion, Supervisor Accuracy, Coordination)

    ---

    ### ⚙️ Configuration

    Set IBM WatsonX credentials in `env.example` (rename to `.env`):
    ```
    IBM_API_KEY=your_api_key
    IBM_PROJECT_ID=your_project_id
    IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
    ```

    > **Note:** Without live credentials, GrantBridge runs in **Simulation Mode** —
    > all agents, guardrails, and evaluation metrics are fully functional using
    > simulated LLM responses.
    """)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    inject_css()
    init_session()
    granite, kb, guardrails, eval_engine = init_core_services()

    page = render_sidebar(granite)

    if page == "🔍 Grant Finder":
        render_grant_finder(granite, kb, guardrails, eval_engine)
    elif page == "📊 LLM Evaluation Dashboard":
        render_evaluation_dashboard(granite, guardrails, eval_engine)
    elif page == "🛡️ Guardrails Monitor":
        render_guardrails_monitor(guardrails)
    elif page == "🤖 Agent Pipeline":
        render_agent_pipeline()
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
