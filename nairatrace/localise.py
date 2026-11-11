"""Fault localisation: which entity caused a failed transaction?

Models see ONLY observable tables (spans, transactions). The hidden ground
truth is used for training labels and scoring, never as a feature. Labels are
the faults the simulator injected, so the task is not circular: a timeout
seen at the switch may have been caused by a slow scheme, a slow issuer or
plain overload, and the model has to tell them apart from evidence.

Scope: technical failures (business declines such as 51 are excluded, since
their cause is in the code itself).

Models:
  rule   - "last responding hop": blame the hop after the deepest one that answered
  gbm    - gradient boosting over per-hop span features and recent-failure-rate
           features per entity (a TraceRCA-style window signal)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import codes

HOP_ORDER = {
    "A": ["TERMINAL", "PTSA", "SWITCH", "SCHEME", "ISSUER", "PRINCIPAL"],
    "B": ["APP", "SENDER_FI", "NIBSS", "BENEFICIARY_FI", "PRINCIPAL"],
    "C": ["APP", "SENDER_FI", "NIBSS", "BENEFICIARY_FI"],
    "D": ["CHECKOUT", "GATEWAY", "SWITCH", "SCHEME", "ISSUER"],
}
ROLES = ["PTSA", "SWITCH", "SCHEME", "ISSUER", "GATEWAY", "SENDER_FI", "NIBSS", "BENEFICIARY_FI", "PRINCIPAL"]
WINDOW_KEYS = ["path_ptsa", "path_switch", "path_scheme", "issuer", "path_gateway",
               "path_sender_fi", "path_beneficiary_fi"]
TECHNICAL_CODES = {(s, c) for s, tbl in codes.SPACES.items() for c, v in tbl.items() if v.kind == "technical"}


def _observably_technical(t):
    """What an observer can tell from codes alone: no code, or a technical code."""
    return t.apply(lambda r: r.final_status != "APPROVED" and (
        pd.isna(r.final_code) or (r.code_space, r.final_code) in TECHNICAL_CODES), axis=1)


def features(result, window_s=600):
    t = result["transactions"].copy()
    s = result["spans"]
    t["obs_technical"] = _observably_technical(t)
    hop = s[s.parent_span_id.notna()]
    agg = hop.groupby(["trace_id", "entity_type"]).agg(
        n=("span_id", "size"), lat=("latency_ms", "max"),
        status=("status", "last"), code=("code", "last")).unstack("entity_type")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    root = s[s.parent_span_id.isna()].set_index("trace_id")[["latency_ms", "status"]]
    root.columns = ["req_latency", "req_status"]
    X = t.set_index("trace_id").join(agg).join(root)
    for r in ROLES:
        for col, fill in ((f"n_{r}", 0), (f"lat_{r}", -1.0), (f"status_{r}", "ABSENT"), (f"code_{r}", "ABSENT")):
            if col not in X:
                X[col] = fill
            X[col] = X[col].fillna(fill)
    # Recent technical-failure rate through the same entity (strictly before this attempt).
    X = X.sort_values("ts")
    for key in WINDOW_KEYS:
        if key not in X:
            X[key] = None
        X[f"win_{key}"] = _window_rate(X, key, window_s)
    X["hour"] = (X.ts % 86400 // 3600).astype(int)
    X["final_code"] = X.final_code.fillna("NONE")
    return X


def _window_rate(X, key, window_s):
    out = pd.Series(np.nan, index=X.index)
    for _, grp in X[X[key].notna()].groupby(key):
        ts = grp.ts.to_numpy()
        bad = grp.obs_technical.to_numpy().astype(float)
        cum = np.concatenate([[0.0], np.cumsum(bad)])
        lo = np.searchsorted(ts, ts - window_s, side="left")
        idx = np.arange(len(ts))
        n = idx - lo
        rate = np.where(n > 0, (cum[idx] - cum[lo]) / np.maximum(n, 1), np.nan)
        out.loc[grp.index] = rate
    return out


def dataset(result):
    X = features(result)
    g = result["ground_truth"].set_index("trace_id")
    X = X.join(g[["root_role", "root_entity", "fault_type", "is_business"]])
    tech = X[X.root_role.notna() & ~X.is_business.astype(bool)]
    return tech


def rule_last_responding_hop(row):
    order = HOP_ORDER[row.route]
    present = [r for r in order[1:] if r in ROLES and row.get(f"n_{r}", 0) > 0]
    if not present:
        return order[1]
    last = present[-1]
    nxt = order[order.index(last) + 1] if order.index(last) + 1 < len(order) else last
    status, code = row[f"status_{last}"], row[f"code_{last}"]
    if status == "ERROR" and code == "96":
        return last
    if status in ("ERROR", "TIMEOUT"):
        return nxt
    if status == "OK":
        # Chain answered OK, so the failure came after it (principal) or before it was seen (terminal).
        return "PRINCIPAL" if order[-1] == "PRINCIPAL" and last != "PRINCIPAL" else order[0]
    return last


CAT_COLS = ["route", "final_code", "final_status", "req_status"] + \
           [f"status_{r}" for r in ROLES] + [f"code_{r}" for r in ROLES]
NUM_COLS = ["req_latency", "hour", "retry_no"] + [f"lat_{r}" for r in ROLES] + \
           [f"n_{r}" for r in ROLES] + [f"win_{k}" for k in WINDOW_KEYS]


def _matrix(df, categories=None):
    M = df[CAT_COLS + NUM_COLS].copy()
    categories = categories or {c: sorted(M[c].astype(str).unique()) for c in CAT_COLS}
    for c in CAT_COLS:
        M[c] = pd.Categorical(M[c].astype(str), categories=categories[c]).codes
    return M, categories


class GBMLocaliser:
    def fit(self, train):
        M, self.categories = _matrix(train)
        self.model = HistGradientBoostingClassifier(
            categorical_features=[M.columns.get_loc(c) for c in CAT_COLS],
            max_iter=300, learning_rate=0.08, random_state=0,
            class_weight="balanced")   # rare roles (PTSA, SCHEME, PRINCIPAL) would otherwise be ignored
        self.model.fit(M, train.root_role)
        return self

    def predict_proba(self, df):
        M, _ = _matrix(df, self.categories)
        return pd.DataFrame(self.model.predict_proba(M), index=df.index, columns=self.model.classes_)


def evaluate(test, proba=None, rule_pred=None):
    """AC@1, AC@3 and macro-F1 over roles (RCAEval-style top-k accuracy)."""
    y = test.root_role
    out = {}
    if rule_pred is not None:
        out["rule"] = {"AC@1": accuracy_score(y, rule_pred), "AC@3": np.nan,
                       "macro_F1": f1_score(y, rule_pred, average="macro", zero_division=0)}
    if proba is not None:
        ranked = np.argsort(-proba.to_numpy(), axis=1)
        cls = proba.columns.to_numpy()
        top1 = cls[ranked[:, 0]]
        top3 = cls[ranked[:, :3]]
        out["gbm"] = {"AC@1": accuracy_score(y, top1),
                      "AC@3": float(np.mean([yy in row for yy, row in zip(y, top3)])),
                      "macro_F1": f1_score(y, top1, average="macro", zero_division=0)}
    majority = y.value_counts().index[0]
    out["majority"] = {"AC@1": float((y == majority).mean()), "AC@3": np.nan,
                       "macro_F1": f1_score(y, [majority] * len(y), average="macro", zero_division=0)}
    return pd.DataFrame(out).T.round(3)


def accuracy_by_fault(test, pred):
    """AC@1 broken down by injected fault type: outages are easy, slow entities are not."""
    df = pd.DataFrame({"fault": test.fault_type, "hit": test.root_role == pred})
    return df.groupby("fault").hit.agg(n="size", AC1="mean").round(3).sort_values("n", ascending=False)


def confusion(test, pred):
    labels = sorted(set(test.root_role) | set(pred))
    return pd.DataFrame(confusion_matrix(test.root_role, pred, labels=labels), index=labels, columns=labels)


def explain(row, proba_row, top=2):
    """One-line, human-readable verdict for a failed transaction."""
    best = proba_row.sort_values(ascending=False).head(top)
    order = HOP_ORDER[row.route]
    evidence = []
    for r in order[1:]:
        if r in ROLES and row.get(f"n_{r}", 0) > 0:
            evidence.append(f"{r} {row[f'status_{r}']}{'/' + str(row[f'code_{r}']) if row[f'code_{r}'] not in (None, 'ABSENT') else ''}"
                            f" {row[f'lat_{r}'] / 1000:.1f}s")
        elif r in ROLES:
            evidence.append(f"no {r} span")
    alt = ", ".join(f"{k} p={v:.2f}" for k, v in best.iloc[1:].items())
    return (f"root={best.index[0]} p={best.iloc[0]:.2f}; evidence: " + "; ".join(evidence)
            + (f"; alternative {alt}" if alt else ""))
