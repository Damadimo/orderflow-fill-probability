"""Validation-day (Sep 1) NLL and C_td of the Cox baselines, for the learning-curve reference line.

Run:  .venv/bin/python scripts/cox_val.py   (writes results/cox_val.json)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines import cox, kaplan_meier  # noqa: E402
from fillprob.data import L2_FEATURES, L3_FEATURES, SPLITS, load_days  # noqa: E402
from fillprob.evaluate import nll_from_surv  # noqa: E402
from fillprob.metrics import antolini_cindex  # noqa: E402

tr = load_days(SPLITS["train"], stride=4, with_tokens=False).df
va = load_days(SPLITS["val"], stride=5, with_tokens=False).df       # same validation subsample as train.py
d, e = va.duration.to_numpy(), va.event.to_numpy()
out = {"n_val": len(va)}
for name, surv in (("B1_kaplan_meier", kaplan_meier(tr, va)),
                   ("B2_cox_L2", cox(tr, va, L2_FEATURES)[0]),
                   ("B2_cox_L2_L3", cox(tr, va, L2_FEATURES + L3_FEATURES)[0])):
    out[name] = {"val_nll": nll_from_surv(surv, d, e), "val_c_td": antolini_cindex(surv, d, e)}
    print(name, out[name], flush=True)
(ROOT / "results" / "cox_val.json").write_text(json.dumps(out, indent=1))
