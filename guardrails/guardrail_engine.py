"""
GrantBridge Guardrails – All 11 guardrail categories.
Each class is independently usable and returns a GuardrailResult.
"""
import re
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    category: str
    message: str = ""
    violations: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed


# ---------------------------------------------------------------------------
# 1. Input Guardrails
# ---------------------------------------------------------------------------

class InputGuardrails:
    """Validate all user inputs before passing to agents."""

    INJECTION_PATTERNS = [
        r"ignore (previous|all|prior) instructions",
        r"you are now",
        r"pretend (you are|to be)",
        r"act as (a |an )?(different|new|evil)",
        r"disregard (your|all) (instructions|guidelines|rules)",
        r"system prompt",
        r"<\|.*?\|>",
        r"\bDAN\b",
        r"jailbreak",
        r"override (safety|guidelines|rules)",
    ]

    PII_PATTERNS = [
        (r"\b\d{10,12}\b", "phone/account number"),
        (r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b", "PAN card"),
        (r"\b\d{12}\b", "Aadhaar-like number"),
        (r"\b[A-Z]{2}\d{6}[A-Z]\b", "passport number"),
        (r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b", "email address"),
    ]

    TOXIC_KEYWORDS = {
        "hate", "kill", "attack", "exploit", "crack", "hack", "phishing",
        "malware", "ransomware", "ddos", "bypass", "steal"
    }

    ALLOWED_FILE_TYPES = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
    MAX_FILE_SIZE_MB = 10

    def validate(self, user_input: str, startup_data: Optional[Dict] = None,
                 file_name: Optional[str] = None, file_size_bytes: Optional[int] = None) -> GuardrailResult:
        violations = []

        # Prompt injection
        inj = self._check_injection(user_input)
        if inj:
            return GuardrailResult(passed=False, category="Input", message="Prompt injection detected.", violations=inj)

        # Toxic / malicious content
        toxic = self._check_toxic(user_input)
        if toxic:
            return GuardrailResult(passed=False, category="Input", message="Malicious content detected.", violations=toxic)

        # PII detection
        pii = self._detect_pii(user_input)
        if pii:
            violations.extend([f"PII detected: {p}" for p in pii])

        # Startup data validation
        if startup_data:
            sv = self._validate_startup_data(startup_data)
            violations.extend(sv)

        # File validation
        if file_name:
            fv = self._validate_file(file_name, file_size_bytes)
            violations.extend(fv)

        # Sanitize
        sanitized = self._sanitize(user_input)

        if violations:
            return GuardrailResult(
                passed=True,  # warnings, not blocks
                category="Input",
                message="Input accepted with warnings.",
                violations=violations,
                sanitized_input=sanitized,
            )
        return GuardrailResult(passed=True, category="Input", message="Input valid.", sanitized_input=sanitized)

    def _check_injection(self, text: str) -> List[str]:
        found = []
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                found.append(f"Injection pattern: {pat[:40]}…")
        return found

    def _check_toxic(self, text: str) -> List[str]:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        return [f"Toxic keyword: {kw}" for kw in (words & self.TOXIC_KEYWORDS)]

    def _detect_pii(self, text: str) -> List[str]:
        found = []
        for pat, label in self.PII_PATTERNS:
            if re.search(pat, text):
                found.append(label)
        return found

    def _validate_startup_data(self, data: Dict) -> List[str]:
        issues = []
        name = data.get("startup_name", "")
        if not name or len(name) < 2:
            issues.append("Startup name too short or missing.")
        industry = data.get("industry", "")
        if not industry:
            issues.append("Industry not specified.")
        location = data.get("location", "")
        if not location:
            issues.append("Location not specified.")
        funding_req = data.get("funding_requirement", 0)
        try:
            if float(funding_req) <= 0:
                issues.append("Funding requirement must be positive.")
        except (ValueError, TypeError):
            issues.append("Invalid funding requirement (must be numeric).")
        return issues

    def _validate_file(self, file_name: str, size_bytes: Optional[int]) -> List[str]:
        issues = []
        ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext not in self.ALLOWED_FILE_TYPES:
            issues.append(f"File type '{ext}' not allowed. Allowed: {', '.join(self.ALLOWED_FILE_TYPES)}")
        if size_bytes and size_bytes > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            issues.append(f"File size exceeds {self.MAX_FILE_SIZE_MB} MB limit.")
        return issues

    def _sanitize(self, text: str) -> str:
        # Strip HTML tags and control characters
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        return text.strip()


# ---------------------------------------------------------------------------
# 2. Output Guardrails
# ---------------------------------------------------------------------------

class OutputGuardrails:
    """Validate AI-generated responses before showing to users."""

    HALLUCINATION_SIGNALS = [
        r"\bI (guarantee|promise|assure)\b",
        r"\b100%\s*(sure|certain|accurate|correct)\b",
        r"\bDefinitely (approved|accepted|guaranteed)\b",
        r"you will (definitely|certainly) (get|receive|win)",
        r"\bno (risk|issue|problem)\b",
    ]

    KNOWN_FAKE_SCHEMES = {
        "pm startup jackpot", "free funding india", "instant grant portal",
        "startup lottery scheme", "guaranteed business loan",
    }

    def validate(self, response_text: str, grants_referenced: Optional[List[Dict]] = None) -> GuardrailResult:
        violations = []

        # Hallucination patterns
        hall = self._check_hallucination(response_text)
        if hall:
            violations.extend(hall)

        # Fake scheme names
        fake = self._check_fake_schemes(response_text)
        if fake:
            violations.extend(fake)

        # Grant name and amount verification
        if grants_referenced:
            gv = self._verify_grant_references(response_text, grants_referenced)
            violations.extend(gv)

        # Sensitive info filter
        si = self._filter_sensitive(response_text)
        if si:
            violations.extend(si)

        # Response format check (if JSON expected)
        if response_text.strip().startswith("{") or response_text.strip().startswith("["):
            fv = self._validate_json_format(response_text)
            if fv:
                violations.extend(fv)

        if violations:
            return GuardrailResult(
                passed=False, category="Output",
                message="Output contains potential issues.",
                violations=violations,
            )
        return GuardrailResult(passed=True, category="Output", message="Output validated.")

    def _check_hallucination(self, text: str) -> List[str]:
        found = []
        for pat in self.HALLUCINATION_SIGNALS:
            if re.search(pat, text, re.IGNORECASE):
                found.append(f"Potential hallucination signal detected.")
        return list(set(found))

    def _check_fake_schemes(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [f"Unverified scheme mentioned: '{scheme}'" for scheme in self.KNOWN_FAKE_SCHEMES
                if scheme in text_lower]

    def _verify_grant_references(self, text: str, grants: List[Dict]) -> List[str]:
        issues = []
        for grant in grants:
            name = grant.get("name", "")
            if name and name.lower() not in text.lower():
                continue
            deadline_str = grant.get("deadline", "")
            if deadline_str:
                try:
                    deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
                    if deadline < datetime.now():
                        issues.append(f"Grant '{name}' deadline may have passed. Verify on official portal.")
                except ValueError:
                    pass
        return issues

    def _filter_sensitive(self, text: str) -> List[str]:
        issues = []
        if re.search(r"api[_-]?key\s*[:=]\s*\S+", text, re.IGNORECASE):
            issues.append("API key exposure detected in output.")
        if re.search(r"password\s*[:=]\s*\S+", text, re.IGNORECASE):
            issues.append("Password exposure detected in output.")
        return issues

    def _validate_json_format(self, text: str) -> List[str]:
        try:
            json.loads(text)
            return []
        except json.JSONDecodeError as exc:
            return [f"Invalid JSON format: {str(exc)[:60]}"]


# ---------------------------------------------------------------------------
# 3. Grant Eligibility Guardrails
# ---------------------------------------------------------------------------

class EligibilityGuardrails:
    """Ensure recommended grants match the startup's actual eligibility."""

    STARTUP_STAGES = ["Idea", "MVP", "Early-stage", "Growth"]

    def validate(self, startup: Dict, grant: Dict) -> GuardrailResult:
        violations = []

        # Stage check
        grant_stages = grant.get("stage", [])
        startup_stage = startup.get("stage", "")
        if grant_stages and startup_stage and startup_stage not in grant_stages:
            violations.append(f"Stage mismatch: Grant requires {grant_stages}, startup is '{startup_stage}'.")

        # Location check
        grant_locations = grant.get("location", [])
        startup_location = startup.get("location", "")
        if grant_locations and "All" not in grant_locations:
            if not any(loc.lower() in startup_location.lower() for loc in grant_locations):
                violations.append(f"Location mismatch: Grant is for {grant_locations}.")

        # Sector check
        grant_sectors = grant.get("sectors", [])
        startup_sector = startup.get("industry", "")
        if grant_sectors and "All" not in grant_sectors:
            if not any(sec.lower() in startup_sector.lower() for sec in grant_sectors):
                violations.append(f"Sector mismatch: Grant targets {grant_sectors}.")

        # Funding amount compatibility
        req_amount = startup.get("funding_requirement", 0)
        if req_amount:
            try:
                req = float(req_amount)
                max_grant = float(grant.get("amount_max", 0))
                if max_grant > 0 and req > max_grant * 1.5:
                    violations.append(
                        f"Funding requirement (${req:,.0f}) far exceeds grant maximum (${max_grant:,.0f})."
                    )
            except (ValueError, TypeError):
                pass

        # Startup age check
        founding_year = startup.get("founding_year")
        if founding_year:
            try:
                age_years = datetime.now().year - int(founding_year)
                if age_years > 10:
                    violations.append("Startup may be too old for some early-stage grants.")
            except (ValueError, TypeError):
                pass

        # DPIIT check (for India-specific grants)
        if any("India" in loc for loc in grant.get("location", [])):
            if not startup.get("dpiit_recognised") and "DPIIT" in grant.get("eligibility", ""):
                violations.append("Grant requires DPIIT recognition, but startup profile does not confirm it.")

        if violations:
            return GuardrailResult(
                passed=False, category="Eligibility",
                message="Startup may not meet all grant eligibility criteria.",
                violations=violations,
            )
        return GuardrailResult(passed=True, category="Eligibility", message="Eligibility criteria met.")


# ---------------------------------------------------------------------------
# 4. RAG / Knowledge Base Guardrails
# ---------------------------------------------------------------------------

class RAGGuardrails:
    """Protect accuracy and reliability of retrieved grant information."""

    MIN_RELEVANCE_SCORE = 0.3
    MAX_INFO_AGE_DAYS = 365

    def validate(self, retrieved_docs: List[Dict], query: str) -> GuardrailResult:
        violations = []
        validated_docs = []

        for doc in retrieved_docs:
            # Relevance check
            score = doc.get("relevance_score", 1.0)
            if score < self.MIN_RELEVANCE_SCORE:
                violations.append(f"Low relevance doc filtered: score={score:.2f}")
                continue

            # Freshness check
            last_verified = doc.get("last_verified", "")
            if last_verified:
                try:
                    lv_date = datetime.strptime(last_verified, "%Y-%m-%d")
                    if (datetime.now() - lv_date).days > self.MAX_INFO_AGE_DAYS:
                        violations.append(f"Outdated grant info: '{doc.get('name', '?')}' last verified {last_verified}.")
                        doc["_outdated_flag"] = True
                except ValueError:
                    pass

            # Source authenticity
            source = doc.get("source", "")
            if source and not source.startswith("https://"):
                violations.append(f"Unverified source URL: {source}")

            # Duplicate detection (by name)
            names = [d.get("name") for d in validated_docs]
            if doc.get("name") in names:
                violations.append(f"Duplicate grant filtered: '{doc.get('name')}'")
                continue

            validated_docs.append(doc)

        if violations:
            return GuardrailResult(
                passed=True,  # warnings – don't fully block, but flag
                category="RAG",
                message="Some retrieved documents flagged.",
                violations=violations,
                metadata={"validated_docs": validated_docs},
            )
        return GuardrailResult(
            passed=True, category="RAG", message="All retrieved documents validated.",
            metadata={"validated_docs": validated_docs},
        )


# ---------------------------------------------------------------------------
# 5. Tool Execution Guardrails
# ---------------------------------------------------------------------------

class ToolExecutionGuardrails:
    """Control how agents interact with external tools and APIs."""

    ALLOWED_DOMAINS = {
        "startupindia.gov.in", "birac.nic.in", "aim.gov.in",
        "meity.gov.in", "seedfund.nsf.gov", "eic.ec.europa.eu",
        "tide.meity.gov.in", "pib.gov.in", "dpiit.gov.in",
    }

    FORBIDDEN_ACTIONS = {
        "delete_grant", "modify_user_profile", "drop_table",
        "delete_record", "admin_override", "system_reset",
    }

    def __init__(self, config):
        self.config = config
        self._call_counts: Dict[str, List[float]] = {}

    def validate_tool_call(self, tool_name: str, params: Dict, agent_id: str) -> GuardrailResult:
        # Forbidden action check
        if tool_name in self.FORBIDDEN_ACTIONS:
            return GuardrailResult(
                passed=False, category="ToolExecution",
                message=f"Action '{tool_name}' is forbidden.",
                violations=[f"Forbidden tool: {tool_name}"],
            )

        # URL/domain validation for web tools
        url = params.get("url", "")
        if url:
            uv = self._validate_url(url)
            if uv:
                return GuardrailResult(passed=False, category="ToolExecution",
                                       message="URL not in approved domain list.", violations=uv)

        # Rate limiting
        rl = self._check_rate_limit(agent_id)
        if rl:
            return GuardrailResult(passed=False, category="ToolExecution",
                                   message="Rate limit exceeded.", violations=rl)

        # SQL injection prevention
        for k, v in params.items():
            if isinstance(v, str) and self._has_sql_injection(v):
                return GuardrailResult(
                    passed=False, category="ToolExecution",
                    message="SQL injection attempt detected.", violations=[f"SQLi in param '{k}'"],
                )

        return GuardrailResult(passed=True, category="ToolExecution", message="Tool call validated.")

    def _validate_url(self, url: str) -> List[str]:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lstrip("www.")
            if not any(domain.endswith(d) for d in self.ALLOWED_DOMAINS):
                return [f"Domain '{domain}' not in approved list."]
        except Exception:
            return ["Invalid URL."]
        return []

    def _check_rate_limit(self, agent_id: str) -> List[str]:
        now = time.time()
        window = 60.0
        calls = [t for t in self._call_counts.get(agent_id, []) if now - t < window]
        calls.append(now)
        self._call_counts[agent_id] = calls
        if len(calls) > self.config.MAX_API_CALLS_PER_MINUTE:
            return [f"Rate limit: {self.config.MAX_API_CALLS_PER_MINUTE} calls/min exceeded."]
        return []

    def _has_sql_injection(self, value: str) -> bool:
        patterns = [r"(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b)", r"(--|;|\/\*|\*\/)", r"('.*'=.*'|1=1)"]
        return any(re.search(p, value, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# 6. Data Guardrails
# ---------------------------------------------------------------------------

class DataGuardrails:
    """Protect startup information and prevent unauthorised data access."""

    def validate_data_access(self, requesting_user_id: str, data_owner_id: str, data_type: str) -> GuardrailResult:
        if requesting_user_id != data_owner_id:
            return GuardrailResult(
                passed=False, category="Data",
                message="Unauthorised data access attempt.",
                violations=[f"User '{requesting_user_id}' cannot access data of '{data_owner_id}'."],
            )
        return GuardrailResult(passed=True, category="Data", message="Data access authorised.")

    def mask_pii(self, data: Dict) -> Dict:
        """Return a copy with sensitive fields masked."""
        sensitive_keys = {"email", "phone", "aadhaar", "pan", "passport", "bank_account", "password"}
        masked = {}
        for k, v in data.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                masked[k] = "***MASKED***"
            else:
                masked[k] = v
        return masked

    def validate_document_upload(self, user_id: str, file_path: str) -> GuardrailResult:
        # Ensure file is stored in user's isolated directory
        expected_prefix = f"./data/uploads/{user_id}/"
        if not file_path.startswith(expected_prefix):
            return GuardrailResult(
                passed=False, category="Data",
                message="Document storage path violation.",
                violations=[f"File must be in user-isolated directory: {expected_prefix}"],
            )
        return GuardrailResult(passed=True, category="Data", message="Document storage validated.")


# ---------------------------------------------------------------------------
# 7. Security Guardrails
# ---------------------------------------------------------------------------

class SecurityGuardrails:
    """Protect GrantBridge against cybersecurity threats."""

    JAILBREAK_PATTERNS = [
        r"you have no (restrictions|limits|rules)",
        r"developer mode",
        r"(unlock|enable) (unrestricted|unsafe|uncensored) mode",
        r"ignore (all |your |the )?(safety|ethical|content) (rules|guidelines|filters)",
        r"respond as (DAN|AIM|STAN|JAILBREAK)",
        r"hypothetically[,.]?\s*if you (were|had) no",
    ]

    def validate(self, text: str) -> GuardrailResult:
        for pat in self.JAILBREAK_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False, category="Security",
                    message="Jailbreak / security threat detected.",
                    violations=[f"Security pattern matched."],
                )
        return GuardrailResult(passed=True, category="Security", message="Security check passed.")

    def validate_jwt_token(self, token: str, secret: str, algorithm: str = "HS256") -> GuardrailResult:
        try:
            import jwt
            payload = jwt.decode(token, secret, algorithms=[algorithm])
            return GuardrailResult(passed=True, category="Security", message="JWT valid.",
                                   metadata={"payload": payload})
        except Exception as exc:
            return GuardrailResult(passed=False, category="Security",
                                   message=f"JWT validation failed: {exc}",
                                   violations=["Invalid or expired JWT token."])


# ---------------------------------------------------------------------------
# 8. Operational Guardrails
# ---------------------------------------------------------------------------

class OperationalGuardrails:
    """Control AI agent execution, API usage and resource consumption."""

    def __init__(self, config):
        self.config = config
        self._execution_start: Dict[str, float] = {}
        self._iteration_counts: Dict[str, int] = {}
        self._previous_outputs: Dict[str, List[str]] = {}

    def start_execution(self, task_id: str):
        self._execution_start[task_id] = time.time()
        self._iteration_counts[task_id] = 0
        self._previous_outputs[task_id] = []

    def check_execution(self, task_id: str, current_output: str = "") -> GuardrailResult:
        # Timeout check
        elapsed = time.time() - self._execution_start.get(task_id, time.time())
        if elapsed > self.config.MAX_EXECUTION_SECONDS:
            return GuardrailResult(
                passed=False, category="Operational",
                message=f"Execution timeout: {elapsed:.0f}s > {self.config.MAX_EXECUTION_SECONDS}s limit.",
                violations=["Execution timeout exceeded."],
            )

        # Max iterations
        self._iteration_counts[task_id] = self._iteration_counts.get(task_id, 0) + 1
        if self._iteration_counts[task_id] > self.config.MAX_AGENT_ITERATIONS:
            return GuardrailResult(
                passed=False, category="Operational",
                message=f"Max iterations exceeded: {self._iteration_counts[task_id]}.",
                violations=["Max agent iterations exceeded."],
            )

        # Loop detection
        prev = self._previous_outputs.get(task_id, [])
        if current_output and current_output in prev[-3:]:
            return GuardrailResult(
                passed=False, category="Operational",
                message="Infinite loop detected: repeated output pattern.",
                violations=["Agent loop detected."],
            )
        if current_output:
            prev.append(current_output)
            self._previous_outputs[task_id] = prev[-10:]

        return GuardrailResult(passed=True, category="Operational", message="Execution within limits.")


# ---------------------------------------------------------------------------
# 9. Human-in-the-Loop Guardrails
# ---------------------------------------------------------------------------

class HumanInTheLoopGuardrails:
    """Ensure human approval for critical actions."""

    HIGH_VALUE_THRESHOLD = 500000  # USD

    APPROVAL_REQUIRED_ACTIONS = {
        "submit_application",
        "send_email",
        "modify_startup_data",
        "delete_startup_data",
        "submit_high_value_grant",
    }

    def requires_approval(self, action: str, context: Optional[Dict] = None) -> GuardrailResult:
        if action in self.APPROVAL_REQUIRED_ACTIONS:
            msg = f"Human approval required for action: '{action}'."
            if context and action == "submit_high_value_grant":
                amount = context.get("amount_max", 0)
                if float(amount) >= self.HIGH_VALUE_THRESHOLD:
                    msg += f" High-value grant (${amount:,.0f}) requires manual eligibility verification."
            return GuardrailResult(
                passed=False, category="HumanInTheLoop",
                message=msg,
                violations=[f"Approval required: {action}"],
                metadata={"requires_human_approval": True, "action": action},
            )
        return GuardrailResult(passed=True, category="HumanInTheLoop", message="Action does not require approval.")


# ---------------------------------------------------------------------------
# 10. Multi-Agent Guardrails
# ---------------------------------------------------------------------------

class MultiAgentGuardrails:
    """Ensure safe coordination between GrantBridge's AI agents."""

    # Which agents can call which tools
    AGENT_PERMISSIONS: Dict[str, List[str]] = {
        "supervisor": ["route_task", "monitor_agents", "approve_output"],
        "grant_discovery": ["search_grants", "scrape_website", "query_knowledge_base"],
        "eligibility": ["check_eligibility", "query_knowledge_base"],
        "rag": ["query_knowledge_base", "embed_document", "retrieve_context"],
        "recommendation": ["generate_recommendation", "format_output"],
    }

    def __init__(self):
        self._handoff_log: List[Dict] = []
        self._output_history: Dict[str, Any] = {}

    def validate_agent_action(self, agent_id: str, action: str) -> GuardrailResult:
        allowed = self.AGENT_PERMISSIONS.get(agent_id, [])
        if action not in allowed:
            return GuardrailResult(
                passed=False, category="MultiAgent",
                message=f"Agent '{agent_id}' is not authorised for action '{action}'.",
                violations=[f"Unauthorised agent action: {agent_id} -> {action}"],
            )
        return GuardrailResult(passed=True, category="MultiAgent", message="Agent action authorised.")

    def validate_handoff(self, from_agent: str, to_agent: str, payload: Dict) -> GuardrailResult:
        log_entry = {
            "from": from_agent, "to": to_agent,
            "timestamp": datetime.now().isoformat(),
            "payload_keys": list(payload.keys()),
        }
        self._handoff_log.append(log_entry)

        # Validate payload completeness
        if not payload:
            return GuardrailResult(
                passed=False, category="MultiAgent",
                message="Empty payload in agent handoff.",
                violations=["Handoff payload is empty."],
            )

        # Cross-agent output verification: check for consistency
        if from_agent in self._output_history:
            prev = self._output_history[from_agent]
            if prev.get("task_id") == payload.get("task_id"):
                # Same task, should have consistent info
                pass

        self._output_history[to_agent] = payload
        return GuardrailResult(
            passed=True, category="MultiAgent", message="Agent handoff validated.",
            metadata={"handoff_log": log_entry},
        )

    @property
    def handoff_log(self) -> List[Dict]:
        return self._handoff_log


# ---------------------------------------------------------------------------
# 11. Ethical & Compliance Guardrails
# ---------------------------------------------------------------------------

class EthicalGuardrails:
    """Ensure responsible and compliant AI behaviour."""

    PROTECTED_ATTRIBUTES = {"gender", "caste", "religion", "race", "ethnicity", "age_group", "disability"}

    def validate(self, recommendation_text: str, startup_data: Dict) -> GuardrailResult:
        violations = []

        # Bias detection: check for protected attribute mentions in ranking logic
        text_lower = recommendation_text.lower()
        for attr in self.PROTECTED_ATTRIBUTES:
            if attr in text_lower:
                violations.append(f"Potential bias: protected attribute '{attr}' referenced in recommendation.")

        # Guarantee language check
        if re.search(r"\b(guarantee|assured|certain|definite(ly)?)\b.*\b(fund|grant|approv)", text_lower):
            violations.append("Recommendation contains unethical guarantee language.")

        # Explainability check: ensure recommendation includes rationale
        if len(recommendation_text) > 50 and "because" not in text_lower and "since" not in text_lower and "based on" not in text_lower:
            violations.append("Recommendation lacks explainability (no rationale provided).")

        # Disclosure of unverified info
        if "source:" not in text_lower and "verified" not in text_lower and "official" not in text_lower:
            violations.append("Recommendation does not cite sources or note verification status.")

        if violations:
            return GuardrailResult(
                passed=True,  # warnings
                category="Ethical",
                message="Ethical review flagged potential issues.",
                violations=violations,
            )
        return GuardrailResult(passed=True, category="Ethical", message="Ethical compliance validated.")

    def generate_audit_trail(self, action: str, agent_id: str, data: Dict) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "data_keys": list(data.keys()),
            "audit_id": f"AUDIT-{int(time.time())}",
        }


# ---------------------------------------------------------------------------
# Unified Guardrail Engine
# ---------------------------------------------------------------------------

class GuardrailEngine:
    """Central engine composing all 11 guardrail categories."""

    def __init__(self, config):
        self.config = config
        self.input = InputGuardrails()
        self.output = OutputGuardrails()
        self.eligibility = EligibilityGuardrails()
        self.rag = RAGGuardrails()
        self.tool_execution = ToolExecutionGuardrails(config)
        self.data = DataGuardrails()
        self.security = SecurityGuardrails()
        self.operational = OperationalGuardrails(config)
        self.human_loop = HumanInTheLoopGuardrails()
        self.multi_agent = MultiAgentGuardrails()
        self.ethical = EthicalGuardrails()
        self._audit_log: List[Dict] = []
        self._stats = {
            "total_checks": 0,
            "passed": 0,
            "blocked": 0,
            "warnings": 0,
            "by_category": {},
        }

    def run_input_checks(self, user_input: str, startup_data: Optional[Dict] = None,
                         file_name: Optional[str] = None, file_size: Optional[int] = None) -> GuardrailResult:
        # Security first
        sec = self.security.validate(user_input)
        if sec.blocked:
            self._record(sec)
            return sec
        result = self.input.validate(user_input, startup_data, file_name, file_size)
        self._record(result)
        return result

    def run_output_checks(self, response: str, grants: Optional[List[Dict]] = None,
                          startup_data: Optional[Dict] = None) -> GuardrailResult:
        out_result = self.output.validate(response, grants)
        if startup_data:
            eth_result = self.ethical.validate(response, startup_data)
            if eth_result.violations:
                out_result.violations.extend(eth_result.violations)
                out_result.passed = out_result.passed and eth_result.passed
        self._record(out_result)
        return out_result

    def _record(self, result: GuardrailResult):
        self._stats["total_checks"] += 1
        if result.blocked:
            self._stats["blocked"] += 1
        elif result.violations:
            self._stats["warnings"] += 1
        else:
            self._stats["passed"] += 1
        cat = result.category
        self._stats["by_category"].setdefault(cat, {"passed": 0, "blocked": 0, "warnings": 0})
        if result.blocked:
            self._stats["by_category"][cat]["blocked"] += 1
        elif result.violations:
            self._stats["by_category"][cat]["warnings"] += 1
        else:
            self._stats["by_category"][cat]["passed"] += 1

    @property
    def stats(self) -> Dict:
        return self._stats
