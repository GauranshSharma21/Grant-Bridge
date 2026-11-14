import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '.')

from core.config import config
from core.granite_client import GraniteClient

granite = GraniteClient(config)
print(f"Model     : {config.GRANITE_MODEL_ID}")
print(f"Live mode : {not granite.is_simulation}")
print()

queries = [
    'Find grants for AI startup in India',
    'Biotech funding opportunities for early stage startup',
    'Government grants for cleantech startup',
]

total_in, total_out = 0, 0
for q in queries:
    r = granite.generate(q)
    total_in += r.input_tokens
    total_out += r.output_tokens
    print(f"Query   : {q}")
    print(f"In tokens: {r.input_tokens}  |  Out tokens: {r.output_tokens}  |  Latency: {round(r.latency_ms)}ms")
    print()

print("=" * 45)
print(f"TOTAL INPUT TOKENS  : {total_in}")
print(f"TOTAL OUTPUT TOKENS : {total_out}")
print(f"TOTAL TOKENS        : {total_in + total_out}")
print(f"TOTAL COST (USD)    : ${granite.usage.total_cost_usd:.6f}")
print("=" * 45)
