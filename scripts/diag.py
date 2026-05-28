# scripts/diag.py — run: python -m scripts.diag
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.rag.generator import generate_answer

# Fire 6 queries back-to-back, fast, like the eval runner does.
# Watch whether later ones come back instantly with empty answers.
queries = [
    ("q2-inflation", "How did rising inflation and input costs affect company margins?", None),
    ("q3-layoffs", "What did management say about hiring freezes, layoffs, or workforce reductions?", None),
    ("q5-competition", "What did companies discuss about competitive pressure or market share losses?", None),
    ("q6-demand", "What was management's outlook on demand for the upcoming quarter?", None),
    ("q7-services", "What did Apple discuss about Services revenue and its growth drivers?", {"ticker": "AAPL"}),
    ("q8-bynd", "What concerns did Beyond Meat management raise about retail sales and consumer demand?", {"ticker": "BYND"}),
]

for name, q, filt in queries:
    t0 = time.perf_counter()
    r = generate_answer(q, k=10, filter=filt)
    dt = time.perf_counter() - t0
    ans = r["answer"]
    print(f"\n[{name}] {dt:.2f}s  abstained={r['abstained']}  n_cit={len(r['citations'])}  ans_len={len(ans)}")
    print(f"   first 200 chars: {ans[:200]!r}")