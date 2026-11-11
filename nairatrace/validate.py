"""Check a run against public Nigerian aggregates (stylised facts).

Targets come from the report's evaluation plan. The failure bands are
snippet-level NIBSS figures [S], so a miss is a prompt to look at the
parameters, not proof the simulator is wrong.
"""

import numpy as np
import pandas as pd

from .world import DAY

TARGETS = {
    "nip_to_pos_volume_ratio": (6.0, 9.5, "NIBSS 2024: NIP ~11.27bn vs POS ~1.45bn txns (~7.7x) [S]"),
    "pos_card_failure_rate": (0.13, 0.15, "NIBSS: POS card failure 13-15% [S]"),
    "nip_failure_rate": (0.005, 0.012, "NIBSS: NIP failure ~0.7% [S]; band widened for peak days"),
    "pos_mean_ticket_naira": (9000, 16000, "NIBSS 2024 POS value/volume ~N12.5k [S]"),
    "failure_dispersion_index": (1.0, np.inf, "Outages are bursty: failures/minute over-dispersed vs Poisson"),
    "share_resolved_after_48h": (0.0, 0.25, "CBN 48h refund rule (2020); a tail beyond it is expected"),
}


def _band(value, lo, hi):
    return "PASS" if lo <= value <= hi else "MISS"


def stylised_facts(result):
    t = result["transactions"]
    first = t[t.retry_no == 0]   # one row per customer intent; retries would double-count volume
    pos = first[first.route == "A"]
    nip = first[first.route.isin(["B", "C"])]
    failed = lambda df: (~df.final_status.isin(["APPROVED"])).mean()

    tech = t.merge(result["ground_truth"], on="trace_id")
    tech = tech[tech.fault_type.notna() & ~tech.is_business]
    horizon = result["params"]["simulation"]["days"] * DAY
    per_min = np.bincount((tech.ts // 60).astype(int), minlength=int(horizon // 60))
    dispersion = per_min.var() / per_min.mean() if per_min.mean() > 0 else np.nan

    resolved = t[t.resolved_at.notna()]
    late = ((resolved.resolved_at - resolved.t_end) > 48 * 3600).mean() if len(resolved) else 0.0

    values = {
        "nip_to_pos_volume_ratio": len(nip) / max(1, len(pos)),
        "pos_card_failure_rate": failed(pos),
        "nip_failure_rate": failed(nip),
        "pos_mean_ticket_naira": pos.amount_kobo.mean() / 100,
        "failure_dispersion_index": dispersion,
        "share_resolved_after_48h": late,
    }
    rows = []
    for k, v in values.items():
        lo, hi, src = TARGETS[k]
        rows.append({"fact": k, "value": round(float(v), 4), "target": f"[{lo}, {hi}]",
                     "check": _band(v, lo, hi), "source": src})
    return pd.DataFrame(rows)


def availability(result):
    """Simulated time-in-fault per role vs A = MTBF / (MTBF + mean MTTR)."""
    p = result["params"]["faults"]
    ep = result["episodes"]
    horizon = result["params"]["simulation"]["days"] * DAY + 3 * DAY
    rows = []
    for role, cfg in p["by_role"].items():
        n_ent = ep[ep.role == role].entity.nunique() if len(ep) else 0
        mean_mttr = cfg["mttr_median_min"] * 60 * np.exp(p["mttr_sigma"] ** 2 / 2)
        expected = cfg["mtbf_h"] * 3600 / (cfg["mtbf_h"] * 3600 + mean_mttr)
        sub = ep[ep.role == role]
        down_time = (sub.t_end - sub.t_start).sum()
        rows.append({"role": role, "episodes": len(sub), "entities_with_faults": n_ent,
                     "expected_availability": round(expected, 4),
                     "fault_time_hours": round(down_time / 3600, 1)})
    return pd.DataFrame(rows)


def code_distribution(result, top=15):
    t = result["transactions"]
    bad = t[t.final_status != "APPROVED"]
    return (bad.assign(code=bad.final_code.fillna("none"))
            .groupby(["route", "code_space", "code"], dropna=False).size()
            .rename("count").reset_index().sort_values("count", ascending=False).head(top))
