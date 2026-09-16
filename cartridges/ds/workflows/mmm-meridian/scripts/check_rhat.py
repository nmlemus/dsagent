"""Auto gate after `fit`: fail unless every R-hat < 1.1 and there are no divergences.

Reads artifacts/diagnostics.json written by the fit step:
  {"rhat_max": 1.03, "divergences": 0, "params": {"roi_m[0]": 1.01, ...}}
"""

import json
import sys
from pathlib import Path

THRESH = 1.1
p = Path("artifacts/diagnostics.json")
if not p.exists():
    sys.exit("diagnostics.json missing — fit step must write it")
d = json.loads(p.read_text())
bad = {k: v for k, v in d.get("params", {}).items() if v >= THRESH}
if d.get("rhat_max", 9) >= THRESH or bad or d.get("divergences", 0) > 0:
    print(f"GATE FAIL: rhat_max={d.get('rhat_max')} divergences={d.get('divergences')} offending={list(bad)[:10]}")
    sys.exit(1)
print(f"GATE PASS: rhat_max={d['rhat_max']} divergences={d.get('divergences', 0)}")
