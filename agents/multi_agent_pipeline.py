"""
GrantBridge Multi-Agent System
Implements Supervisor, Grant Discovery, Eligibility, RAG, and Recommendation agents
using a LangGraph-inspired state machine (pure Python fallback for portability).
"""
import time
import logging
import hashlib
import uuid
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class AgentMetrics:
    agent_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    handoffs_sent: int = 0
    handoffs_received: int = 0
    reasoning_steps: List[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return (end - self.start_time) * 1000

    @property
    def tool_call_accuracy(self) -> float:
        if self.tool_calls == 0:
            return 1.0
        return self.tool_successes / self.tool_calls

    @property
    def task_completion_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total


@dataclass
class WorkflowState:
    task_id: str
    startup_profile: Dict[str, Any]
    user_query: str
    status: AgentStatus = AgentStatus.PENDING
    current_agent: str = "supervisor"
    discovered_grants: List[Dict] = field(default_factory=list)
    eligible_grants: List[Dict] = field(default_factory=list)
    retrieved_context: List[Dict] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    final_response: str = ""
    errors: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False
    human_approved: bool = False
    iterations: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, agent_id: str, granite_client, guardrail_engine, knowledge_base):
        self.agent_id = agent_id
        self.llm = granite_client
        self.guardrails = guardrail_engine
        self.kb = knowledge_base
        self.metrics = AgentMetrics(agent_id=agent_id)
        self._log: List[Dict] = []

    def _log_step(self, step: str, details: str = ""):
        entry = {"step": step, "details": details, "timestamp": datetime.now().isoformat()}
        self._log.append(entry)
        self.metrics.reasoning_steps.append(f"{step}: {details[:80]}")
        logger.debug(f"[{self.agent_id}] {step}: {details[:100]}")

    def _tool_call(self, tool_name: str, params: Dict, agent_id: Optional[str] = None) -> Dict:
        aid = agent_id or self.agent_id
        self.metrics.tool_calls += 1
        result = self.guardrails.tool_execution.validate_tool_call(tool_name, params, aid)
        if result.blocked:
            self._log_step("tool_blocked", f"{tool_name}: {result.message}")
            self.metrics.tasks_failed += 1
            return {"success": False, "error": result.message}
        self.metrics.tool_successes += 1
        return {"success": True}

    def run(self, state: WorkflowState) -> WorkflowState:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Supervisor Agent
# ---------------------------------------------------------------------------

class SupervisorAgent(BaseAgent):
    """Routes tasks, monitors coordination, approves outputs."""

    ROUTE_PLAN = ["grant_discovery", "eligibility", "rag", "recommendation"]

    def __init__(self, *args, **kwargs):
        super().__init__("supervisor", *args, **kwargs)
        self._agent_registry: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        self._agent_registry[agent.agent_id] = agent

    def run(self, state: WorkflowState) -> WorkflowState:
        self.metrics.start_time = time.time()
        self._log_step("supervisor_start", f"task_id={state.task_id}")

        # Validate handoff to first agent
        handoff_check = self.guardrails.multi_agent.validate_handoff(
            "user", "supervisor",
            {"task_id": state.task_id, "query": state.user_query},
        )
        if handoff_check.blocked:
            state.errors.append(handoff_check.message)
            state.status = AgentStatus.FAILED
            return state

        # Operational guard: start tracking
        self.guardrails.operational.start_execution(state.task_id)

        # Execute pipeline
        for agent_id in self.ROUTE_PLAN:
            exec_check = self.guardrails.operational.check_execution(state.task_id)
            if exec_check.blocked:
                state.errors.append(exec_check.message)
                break

            agent = self._agent_registry.get(agent_id)
            if not agent:
                state.errors.append(f"Agent '{agent_id}' not registered.")
                continue

            # Validate handoff
            hv = self.guardrails.multi_agent.validate_handoff(
                "supervisor", agent_id, {"task_id": state.task_id}
            )
            if hv.blocked:
                state.errors.append(hv.message)
                continue

            self._log_step("dispatch", f"→ {agent_id}")
            state.current_agent = agent_id
            state.iterations += 1
            try:
                state = agent.run(state)
                self.metrics.handoffs_sent += 1
            except Exception as exc:
                state.errors.append(f"{agent_id} error: {str(exc)[:100]}")
                logger.error(f"Agent {agent_id} raised: {exc}")

        state.current_agent = "supervisor"
        # Human-in-the-loop check for application submission
        if state.recommendations:
            hitl = self.guardrails.human_loop.requires_approval("submit_application")
            if hitl.blocked:
                state.requires_human_approval = True
                state.status = AgentStatus.AWAITING_APPROVAL
            else:
                state.status = AgentStatus.COMPLETED
        else:
            state.status = AgentStatus.COMPLETED

        self.metrics.end_time = time.time()
        self.metrics.tasks_completed += 1
        self._log_step("supervisor_complete", f"status={state.status.value}")
        return state


# ---------------------------------------------------------------------------
# Grant Discovery Agent
# ---------------------------------------------------------------------------

class GrantDiscoveryAgent(BaseAgent):
    """Finds relevant funding opportunities from knowledge base and web sources."""

    def __init__(self, *args, **kwargs):
        super().__init__("grant_discovery", *args, **kwargs)

    def run(self, state: WorkflowState) -> WorkflowState:
        self.metrics.start_time = time.time()
        self._log_step("discovery_start", state.user_query[:80])

        # Validate agent action
        action_check = self.guardrails.multi_agent.validate_agent_action(
            "grant_discovery", "query_knowledge_base"
        )
        if action_check.blocked:
            state.errors.append(action_check.message)
            self.metrics.tasks_failed += 1
            return state

        # Build search query from startup profile
        profile = state.startup_profile
        search_query = (
            f"{profile.get('industry', '')} {profile.get('stage', '')} "
            f"{profile.get('location', '')} {state.user_query}"
        ).strip()

        self._log_step("kb_search", search_query[:60])
        tool_result = self._tool_call("query_knowledge_base", {"query": search_query})
        if not tool_result["success"]:
            state.errors.append(f"KB search blocked: {tool_result['error']}")
            self.metrics.tasks_failed += 1
            return state

        grants = self.kb.search(search_query, n_results=10)
        self._log_step("grants_found", f"{len(grants)} results from KB")

        # LLM enhancement: generate supplementary insights
        prompt = (
            f"You are a grant discovery expert. A startup in the {profile.get('industry', 'technology')} sector "
            f"at {profile.get('stage', 'Early-stage')} stage located in {profile.get('location', 'India')} "
            f"with funding requirement of {profile.get('funding_requirement', '50000')} USD is looking for grants.\n"
            f"User query: {state.user_query}\n\n"
            f"Based on these retrieved grants:\n"
            + "\n".join([f"- {g.get('name', 'Unknown')}" for g in grants[:5]])
            + "\n\nProvide a brief analysis of the top 3 most relevant funding opportunities."
        )
        llm_result = self.llm.generate(prompt, max_tokens=512)
        self._log_step("llm_analysis", f"{llm_result.output_tokens} tokens")

        state.discovered_grants = grants
        state.agent_outputs["grant_discovery"] = {
            "grants_found": len(grants),
            "llm_analysis": llm_result.text,
            "latency_ms": self.metrics.latency_ms,
        }
        self.metrics.end_time = time.time()
        self.metrics.tasks_completed += 1
        return state


# ---------------------------------------------------------------------------
# Eligibility Agent
# ---------------------------------------------------------------------------

class EligibilityAgent(BaseAgent):
    """Verifies which discovered grants the startup is eligible for."""

    def __init__(self, *args, **kwargs):
        super().__init__("eligibility", *args, **kwargs)

    def run(self, state: WorkflowState) -> WorkflowState:
        self.metrics.start_time = time.time()
        self._log_step("eligibility_start", f"{len(state.discovered_grants)} grants to check")

        eligible = []
        ineligible_reasons = []
        profile = state.startup_profile

        for grant in state.discovered_grants:
            action_check = self.guardrails.multi_agent.validate_agent_action(
                "eligibility", "check_eligibility"
            )
            if action_check.blocked:
                break

            tool_result = self._tool_call("check_eligibility", {"grant_id": grant.get("id", "")})
            if not tool_result["success"]:
                continue

            elig_result = self.guardrails.eligibility.validate(profile, grant)
            if elig_result.passed and not elig_result.blocked:
                grant["eligibility_status"] = "eligible"
                grant["eligibility_notes"] = elig_result.message
                eligible.append(grant)
                self._log_step("eligible", grant.get("name", "?")[:50])
            else:
                grant["eligibility_status"] = "ineligible"
                grant["eligibility_violations"] = elig_result.violations
                ineligible_reasons.append({
                    "grant": grant.get("name"), "reasons": elig_result.violations
                })
                self._log_step("ineligible", f"{grant.get('name', '?')[:40]}: {elig_result.violations[:1]}")

        # LLM eligibility summary
        prompt = (
            f"As a grant eligibility expert, assess why the following grants are suitable for this startup:\n"
            f"Startup: {profile.get('startup_name', 'N/A')}, "
            f"Industry: {profile.get('industry', 'N/A')}, "
            f"Stage: {profile.get('stage', 'N/A')}, "
            f"Location: {profile.get('location', 'N/A')}\n\n"
            f"Eligible grants: {[g.get('name') for g in eligible]}\n"
            f"Provide brief eligibility justification."
        )
        llm_result = self.llm.generate(prompt, max_tokens=400)

        state.eligible_grants = eligible
        state.agent_outputs["eligibility"] = {
            "eligible_count": len(eligible),
            "ineligible_count": len(state.discovered_grants) - len(eligible),
            "ineligible_reasons": ineligible_reasons,
            "llm_summary": llm_result.text,
            "latency_ms": self.metrics.latency_ms,
        }
        self.metrics.end_time = time.time()
        self.metrics.tasks_completed += 1
        return state


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

class RAGAgent(BaseAgent):
    """Retrieves and validates contextual grant information."""

    def __init__(self, *args, **kwargs):
        super().__init__("rag", *args, **kwargs)

    def run(self, state: WorkflowState) -> WorkflowState:
        self.metrics.start_time = time.time()
        self._log_step("rag_start", f"{len(state.eligible_grants)} eligible grants")

        # Build focused retrieval query
        grant_names = [g.get("name", "") for g in state.eligible_grants[:3]]
        query = f"{state.user_query} {' '.join(grant_names)}"

        tool_result = self._tool_call("retrieve_context", {"query": query})
        if not tool_result["success"]:
            state.retrieved_context = state.eligible_grants
            self.metrics.end_time = time.time()
            return state

        retrieved = self.kb.search(query, n_results=5)
        rag_check = self.guardrails.rag.validate(retrieved, query)

        validated_docs = rag_check.metadata.get("validated_docs", retrieved)
        self._log_step("rag_retrieved", f"{len(validated_docs)} validated docs")

        # Context relevance via LLM
        if validated_docs:
            context_text = "\n\n".join([
                d.get("document_text", self.kb._build_document_text(d))
                for d in validated_docs[:3]
            ])
            prompt = (
                f"Based on the following grant information:\n{context_text[:1500]}\n\n"
                f"Answer this query: {state.user_query}\n"
                f"Provide verified, source-cited information only."
            )
            llm_result = self.llm.generate(prompt, max_tokens=600)
            self._log_step("rag_generation", f"{llm_result.output_tokens} tokens")
        else:
            llm_result = self.llm.generate(state.user_query, max_tokens=400)

        state.retrieved_context = validated_docs
        state.agent_outputs["rag"] = {
            "docs_retrieved": len(retrieved),
            "docs_validated": len(validated_docs),
            "rag_violations": rag_check.violations,
            "grounded_response": llm_result.text,
            "latency_ms": self.metrics.latency_ms,
        }
        self.metrics.end_time = time.time()
        self.metrics.tasks_completed += 1
        return state


# ---------------------------------------------------------------------------
# Recommendation Agent
# ---------------------------------------------------------------------------

class RecommendationAgent(BaseAgent):
    """Generates final, ranked grant recommendations."""

    def __init__(self, *args, **kwargs):
        super().__init__("recommendation", *args, **kwargs)

    def run(self, state: WorkflowState) -> WorkflowState:
        self.metrics.start_time = time.time()
        self._log_step("recommendation_start", f"{len(state.eligible_grants)} candidates")

        grants_to_recommend = state.eligible_grants or state.discovered_grants
        profile = state.startup_profile

        # Build recommendation prompt
        grant_summaries = []
        for g in grants_to_recommend[:5]:
            summary = (
                f"- **{g.get('name', 'Unknown')}**: "
                f"{g.get('currency', 'USD')} {g.get('amount_min', 0):,} – {g.get('amount_max', 0):,}. "
                f"Deadline: {g.get('deadline', 'Unknown')}. "
                f"Source: {g.get('source', 'N/A')}."
            )
            grant_summaries.append(summary)

        context = "\n".join(grant_summaries) if grant_summaries else "No specific grants retrieved."
        rag_context = state.agent_outputs.get("rag", {}).get("grounded_response", "")

        prompt = (
            f"You are GrantBridge, an AI grant recommendation system.\n\n"
            f"Startup Profile:\n"
            f"- Name: {profile.get('startup_name', 'N/A')}\n"
            f"- Industry: {profile.get('industry', 'N/A')}\n"
            f"- Stage: {profile.get('stage', 'N/A')}\n"
            f"- Location: {profile.get('location', 'N/A')}\n"
            f"- Funding Requirement: {profile.get('funding_requirement', 'N/A')} USD\n"
            f"- Description: {profile.get('description', 'N/A')}\n\n"
            f"Verified Grant Opportunities:\n{context}\n\n"
            f"Additional Context:\n{rag_context[:800]}\n\n"
            f"User Query: {state.user_query}\n\n"
            f"Provide personalised grant recommendations with:\n"
            f"1. Grant name and funding amount\n"
            f"2. Why this grant matches the startup\n"
            f"3. Key application requirements\n"
            f"4. Next steps\n"
            f"Note: Mark all amounts and deadlines as approximate. Recommend official portal verification."
        )

        llm_result = self.llm.generate(prompt, max_tokens=800)
        self._log_step("recommendation_generated", f"{llm_result.output_tokens} tokens")

        # Output guardrails
        out_check = self.guardrails.run_output_checks(
            llm_result.text, grants_to_recommend, profile
        )
        if out_check.blocked:
            self._log_step("output_blocked", str(out_check.violations[:2]))
            state.final_response = (
                "⚠️ Output guardrails blocked the recommendation. "
                "Please try a more specific query or contact support."
            )
            self.metrics.tasks_failed += 1
        else:
            # Build structured recommendations
            recommendations = []
            for g in grants_to_recommend[:5]:
                rec = {
                    "id": g.get("id", str(uuid.uuid4())[:8]),
                    "name": g.get("name", "Unknown"),
                    "amount_min": g.get("amount_min", 0),
                    "amount_max": g.get("amount_max", 0),
                    "currency": g.get("currency", "USD"),
                    "deadline": g.get("deadline", "Unknown"),
                    "source": g.get("source", ""),
                    "eligibility_status": g.get("eligibility_status", "pending"),
                    "relevance_score": g.get("relevance_score", 0.75),
                    "match_reason": f"Matches {profile.get('industry')} sector and {profile.get('stage')} stage.",
                    "sectors": g.get("sectors", []),
                    "stage": g.get("stage", []),
                    "verified": g.get("verified", False),
                }
                recommendations.append(rec)

            state.recommendations = recommendations
            state.final_response = llm_result.text
            self.metrics.tasks_completed += 1

        state.agent_outputs["recommendation"] = {
            "grants_recommended": len(state.recommendations),
            "output_violations": out_check.violations,
            "latency_ms": self.metrics.latency_ms,
            "tokens_used": llm_result.output_tokens,
        }
        self.metrics.end_time = time.time()
        return state


# ---------------------------------------------------------------------------
# Multi-Agent Pipeline Factory
# ---------------------------------------------------------------------------

def build_pipeline(granite_client, guardrail_engine, knowledge_base):
    """Build and wire all agents into a supervised pipeline."""
    supervisor = SupervisorAgent(granite_client, guardrail_engine, knowledge_base)
    discovery = GrantDiscoveryAgent(granite_client, guardrail_engine, knowledge_base)
    eligibility = EligibilityAgent(granite_client, guardrail_engine, knowledge_base)
    rag = RAGAgent(granite_client, guardrail_engine, knowledge_base)
    recommendation = RecommendationAgent(granite_client, guardrail_engine, knowledge_base)

    for agent in [discovery, eligibility, rag, recommendation]:
        supervisor.register_agent(agent)

    return supervisor, {
        "supervisor": supervisor,
        "grant_discovery": discovery,
        "eligibility": eligibility,
        "rag": rag,
        "recommendation": recommendation,
    }


def run_grant_search(
    startup_profile: Dict,
    user_query: str,
    granite_client,
    guardrail_engine,
    knowledge_base,
) -> WorkflowState:
    """Entry point for the grant search workflow."""
    task_id = f"TASK-{hashlib.md5(f'{user_query}{time.time()}'.encode()).hexdigest()[:8].upper()}"
    state = WorkflowState(
        task_id=task_id,
        startup_profile=startup_profile,
        user_query=user_query,
    )

    supervisor, agents = build_pipeline(granite_client, guardrail_engine, knowledge_base)
    state = supervisor.run(state)
    return state, agents
