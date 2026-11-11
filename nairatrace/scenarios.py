"""Named scenarios: parameter overrides applied on top of default.yaml.

`stress` is not meant to be realistic. It raises infrastructure fault rates
and the share of DEGRADED (slow, not dead) faults so that every entity role
fails often enough to train and test a localiser, including the hard cases
where a timeout at one hop was caused by a slow hop further down.
"""

from .params import load


def overrides(name, base_path=None):
    if name in (None, "default"):
        return {}
    if name == "stress":
        p, _ = load() if base_path is None else load(base_path)
        by_role = {}
        for role, cfg in p["faults"]["by_role"].items():
            by_role[role] = {**cfg, "mtbf_h": cfg["mtbf_h"] / 6, "p_down": 0.3}
        return {"faults.by_role": by_role, "faults.degraded_latency_x": 60.0}
    raise ValueError(f"unknown scenario {name!r}")
