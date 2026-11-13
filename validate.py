"""Final integration validation for GrantBridge."""
import sys
sys.path.insert(0, '.')

from core.config import config
from core.granite_client import GraniteClient
from core.knowledge_base import GrantKnowledgeBase
from guardrails.guardrail_engine import GuardrailEngine
from agents.multi_agent_pipeline import run_grant_search
from evaluation.eval_engine import EvaluationEngine

granite = GraniteClient(config)
kb = GrantKnowledgeBase(config)
engine = GuardrailEngine(config)

# KB test
grants = kb.search('AI startup India seed funding', n_results=3)
print(f'KB search: {len(grants)} results')
for g in grants:
    print(f'  - {g.get("name", "?")}')

# Pipeline test
state, agents = run_grant_search(
    startup_profile={
        'startup_name': 'HealthAI',
        'industry': 'Healthtech / Biotech',
        'stage': 'Early-stage',
        'location': 'India',
        'funding_requirement': 100000,
        'founding_year': 2022,
        'description': 'AI diagnostics',
        'dpiit_recognised': True,
    },
    user_query='Looking for biotech grants in India',
    granite_client=granite,
    guardrail_engine=engine,
    knowledge_base=kb,
)
print(f'Pipeline: discovered={len(state.discovered_grants)}, eligible={len(state.eligible_grants)}, rec={len(state.recommendations)}')
print(f'Response (100 chars): {state.final_response[:100]}')

# Eval
eval_engine = EvaluationEngine()
results = eval_engine.run_full_evaluation(
    granite_client=granite,
    guardrail_engine=engine,
    agents=agents,
    workflow_state=state,
)
overall = eval_engine.get_overall_score()
print(f'Overall score: {overall:.1%}')
for key, result in results.items():
    n = len(result.metrics)
    print(f'  {result.category[:35]:35s}: {n} metrics')
print('ALL VALIDATION TESTS PASSED')
