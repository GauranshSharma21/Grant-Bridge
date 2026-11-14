# 🌉 GrantBridge — AI Grant & Funding Finder for Startups

> **Problem Statement #18 · IBM SkillsBuild · AICTE Project**

GrantBridge is an AI-powered Grant and Funding Finder for startups, built using **IBM Granite 4** via WatsonX AI, a **multi-agent LangGraph pipeline**, **ChromaDB** for RAG, and **Streamlit** for the interactive UI.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🏗️ Architecture

```
User Input
    │
Input Guardrails (PII · Injection · Validation)
    │
Supervisor Agent (LangGraph)
    ├── Grant Discovery Agent  → ChromaDB + IBM Granite
    ├── Eligibility Agent      → Criteria validation
    ├── RAG Agent              → Context retrieval
    └── Recommendation Agent  → Ranked results
    │
Output Guardrails (Hallucination · Accuracy · Source)
    │
Human-in-the-Loop Approval
    │
GrantBridge Dashboard
```

---

## 🤖 Multi-Agent System

| Agent | Role |
|---|---|
| **Supervisor Agent** | Routes tasks, monitors coordination, enforces HITL |
| **Grant Discovery Agent** | Searches ChromaDB knowledge base + IBM Granite analysis |
| **Eligibility Agent** | Validates startup eligibility against grant criteria |
| **RAG Agent** | Retrieves and validates contextual grant information |
| **Recommendation Agent** | Generates ranked, personalised recommendations |

---

## 🛡️ 11 Guardrail Categories

1. Input Guardrails — PII detection, prompt injection, validation
2. Output Guardrails — Hallucination detection, factual accuracy
3. Eligibility Guardrails — Stage, sector, location matching
4. RAG & Knowledge Base Guardrails — Freshness, relevance, source verification
5. Tool Execution Guardrails — Rate limiting, URL validation, SQL injection prevention
6. Data Guardrails — Privacy, masking, user data isolation
7. Security Guardrails — Jailbreak detection, JWT auth
8. Operational Guardrails — Timeout, loop detection, budget limits
9. Human-in-the-Loop Guardrails — Approval before submission
10. Multi-Agent Guardrails — Handoff validation, coordination
11. Ethical & Compliance Guardrails — Bias detection, audit trail

---

## 📊 LLM Evaluation Dashboard — 47 Metrics

| Category | Metrics |
|---|---|
| Accuracy & Quality | Accuracy, Precision, Recall, F1, ROUGE-L, BLEU, Semantic Similarity, Exact Match |
| RAG Evaluation | Context Precision, Recall, Faithfulness, Groundedness, Answer Relevancy |
| Agentic AI | Task Completion, Tool Accuracy, Handoff, Loop Detection, Planning, Reasoning |
| Safety & Guardrails | Hallucination Rate, Toxicity, Bias, PII Leakage, Guardrail Intervention Rate |
| Performance & Cost | Latency, TTFT, Tokens/sec, Cost/Request, Error Rate, Throughput |
| Multi-Agent System | E2E Completion, Supervisor Accuracy, Inter-Agent Consistency, Distribution Efficiency |

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| LLM | IBM Granite 4 (`ibm/granite-4-h-small`) via WatsonX AI |
| Agent Orchestration | LangGraph / LangChain |
| Vector Database | ChromaDB |
| Document Processing | PyMuPDF |
| Web Scraping | BeautifulSoup4 |
| Frontend | Streamlit |
| Cloud Platform | IBM Cloud Lite |

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/GauranshSharma21/Grant-Bridge.git
cd Grant-Bridge
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp env.example .env
```
Edit `.env` and add your IBM credentials:
```
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_ibm_project_id
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-4-h-small
```

### 4. Run the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🌐 Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repository → `main` branch → `app.py`
4. Add your IBM credentials in **Advanced Settings → Secrets**
5. Click **Deploy**

---

## 📁 Project Structure

```
Grant-Bridge/
├── app.py                      # Main Streamlit application
├── env.example                 # Environment variable template
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies
├── app.json                    # Deployment configuration
├── run.bat                     # Windows quick-start script
├── core/
│   ├── config.py               # Configuration management
│   ├── granite_client.py       # IBM Granite LLM client
│   └── knowledge_base.py       # ChromaDB RAG knowledge base
├── agents/
│   └── multi_agent_pipeline.py # 5-agent LangGraph pipeline
├── guardrails/
│   └── guardrail_engine.py     # All 11 guardrail categories
├── evaluation/
│   └── eval_engine.py          # LLM evaluation engine (47 metrics)
└── data/
    ├── grants/                 # Grant data files
    └── uploads/                # User uploaded documents
```

---

## 🔑 IBM Granite Token Usage

| Metric | Value |
|---|---|
| Model | `ibm/granite-4-h-small` |
| Tokens per grant search | ~600 tokens |
| Average latency | ~940ms |
| Cost per request | ~$0.000089 |
| Free tier | 50,000 tokens/month (IBM Cloud Lite) |

---

## 📄 License

MIT License — built for IBM SkillsBuild Problem Statement #18.
