"""
IBM Granite LLM Client
Wraps IBM WatsonX AI with fallback simulation for demo/testing.
"""
import time
import random
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    finish_reason: str = "stop"
    cost_usd: float = 0.0


@dataclass
class UsageStats:
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    failed_requests: int = 0
    total_cost_usd: float = 0.0
    request_times: List[float] = field(default_factory=list)


class GraniteClient:
    """
    IBM Granite LLM client with real WatsonX integration and
    simulation fallback when credentials are not configured.
    """

    # Approximate pricing (USD per 1k tokens)
    INPUT_COST_PER_1K = 0.0005
    OUTPUT_COST_PER_1K = 0.0015

    def __init__(self, config):
        self.config = config
        self.usage = UsageStats()
        self._client = None
        self._use_simulation = True
        self._init_client()

    def _init_client(self):
        """Attempt to initialise WatsonX AI client using the modern chat endpoint."""
        if not self.config.has_ibm_credentials():
            logger.warning("IBM credentials not configured – using simulation mode.")
            return
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference

            credentials = Credentials(
                url=self.config.IBM_WATSONX_URL,
                api_key=self.config.IBM_API_KEY,
            )
            self._client = ModelInference(
                model_id=self.config.GRANITE_MODEL_ID,
                credentials=credentials,
                project_id=self.config.IBM_PROJECT_ID,
            )
            # Verify connectivity using the modern chat endpoint
            test = self._client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                params={"max_tokens": 5},
            )
            self._use_simulation = False
            logger.info(f"IBM WatsonX Granite (chat) initialised: {self.config.GRANITE_MODEL_ID}")
        except Exception as exc:
            logger.warning(f"WatsonX init failed ({exc}) – falling back to simulation.")

    def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """Generate a response from Granite (or simulate one)."""
        start = time.time()
        input_tokens = max(1, len(prompt.split()) * 4 // 3)

        if not self._use_simulation and self._client:
            return self._real_generate(prompt, max_tokens, start, input_tokens)
        return self._simulate_generate(prompt, max_tokens, start, input_tokens)

    def _real_generate(self, prompt: str, max_tokens: int, start: float, input_tokens: int) -> LLMResponse:
        try:
            response = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                params={"max_tokens": min(max_tokens, 1024), "temperature": 0.3, "top_p": 0.9},
            )
            latency_ms = (time.time() - start) * 1000
            # Extract text from chat response structure
            choices = response.get("choices", [])
            result = choices[0]["message"]["content"] if choices else ""
            # Use token counts from API if available
            usage = response.get("usage", {})
            actual_input = usage.get("prompt_tokens", input_tokens)
            actual_output = usage.get("completion_tokens", max(1, len(result.split()) * 4 // 3))
            cost = (actual_input / 1000 * self.INPUT_COST_PER_1K +
                    actual_output / 1000 * self.OUTPUT_COST_PER_1K)
            self._record(actual_input, actual_output, latency_ms, cost)
            return LLMResponse(
                text=result,
                input_tokens=actual_input,
                output_tokens=actual_output,
                latency_ms=latency_ms,
                model_id=self.config.GRANITE_MODEL_ID,
                cost_usd=cost,
            )
        except Exception as exc:
            self.usage.failed_requests += 1
            logger.error(f"WatsonX generation error: {exc}")
            return self._simulate_generate(prompt, max_tokens, time.time(), input_tokens)

    def _simulate_generate(self, prompt: str, max_tokens: int, start: float, input_tokens: int) -> LLMResponse:
        """Deterministic simulation producing realistic-looking grant responses."""
        time.sleep(random.uniform(0.3, 0.9))  # simulate network latency
        latency_ms = (time.time() - start) * 1000
        output_tokens = random.randint(80, min(max_tokens, 400))

        text = self._build_simulated_response(prompt)
        cost = (input_tokens / 1000 * self.INPUT_COST_PER_1K +
                output_tokens / 1000 * self.OUTPUT_COST_PER_1K)
        self._record(input_tokens, output_tokens, latency_ms, cost)
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_id=self.config.GRANITE_MODEL_ID + " (simulation)",
            cost_usd=cost,
        )

    def _build_simulated_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "eligibility" in prompt_lower:
            return (
                "Based on the startup profile provided, the following eligibility criteria apply:\n"
                "1. The startup must be registered under DPIIT / relevant national authority.\n"
                "2. Annual revenue must not exceed the grant ceiling.\n"
                "3. The business must operate in an eligible sector as defined by the scheme.\n"
                "4. At least one co-founder must be a citizen of the host country.\n"
                "Recommendation: Verify DPIIT certificate and turnover certificate before applying."
            )
        if "grant" in prompt_lower or "funding" in prompt_lower:
            return (
                "Top matching grants for your startup:\n\n"
                "1. **Startup India Seed Fund Scheme** – Up to ₹20 lakhs for proof-of-concept and prototype development.\n"
                "   Deadline: Rolling applications. Eligibility: DPIIT-recognised startups < 2 years.\n\n"
                "2. **BIRAC BIG Grant** – Up to ₹50 lakhs for biotech/healthtech innovations.\n"
                "   Deadline: Quarterly calls. Eligibility: Startups with strong IP component.\n\n"
                "3. **Atal Innovation Mission (AIM) Grants** – Up to ₹10 crores for deep-tech startups.\n"
                "   Deadline: Annual cycle. Eligibility: DPIIT-recognised, revenue < ₹25 crores.\n\n"
                "⚠️ All amounts are approximate. Verify deadlines on official portals before applying."
            )
        if "proposal" in prompt_lower or "application" in prompt_lower:
            return (
                "**Draft Grant Application**\n\n"
                "Executive Summary: This startup addresses a critical market gap through innovative technology...\n\n"
                "Problem Statement: [Describe the problem your startup solves]\n\n"
                "Proposed Solution: [Describe your solution, technology, and approach]\n\n"
                "Market Opportunity: The addressable market is estimated at $X billion...\n\n"
                "Team: [Describe founder qualifications and experience]\n\n"
                "Financials: Requested funding of [Amount] will be deployed as follows: 40% R&D, 30% market expansion, 30% operations.\n\n"
                "⚠️ This draft requires human review and verification before submission."
            )
        return (
            "I have analysed the request and identified the following key insights for your startup funding journey. "
            "Based on your profile, there are several relevant funding opportunities available. "
            "I recommend verifying all grant details on official government portals before proceeding with applications."
        )

    def _record(self, input_tokens: int, output_tokens: int, latency_ms: float, cost: float):
        self.usage.total_requests += 1
        self.usage.total_input_tokens += input_tokens
        self.usage.total_output_tokens += output_tokens
        self.usage.total_latency_ms += latency_ms
        self.usage.total_cost_usd += cost
        self.usage.request_times.append(time.time())

    @property
    def avg_latency_ms(self) -> float:
        if self.usage.total_requests == 0:
            return 0.0
        return self.usage.total_latency_ms / self.usage.total_requests

    @property
    def error_rate(self) -> float:
        total = self.usage.total_requests + self.usage.failed_requests
        if total == 0:
            return 0.0
        return self.usage.failed_requests / total

    @property
    def is_simulation(self) -> bool:
        return self._use_simulation
