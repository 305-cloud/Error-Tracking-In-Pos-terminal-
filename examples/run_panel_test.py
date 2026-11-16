"""First test of the fault localiser on the 8-person panel.

1. Rebuild the panel world (seed 8, 5 days, 8 people at 8 different banks).
2. Train the gradient-boosting localiser on OTHER simulated worlds (different seeds),
   so nothing from the panel world is seen in training.
3. For every panel failure:
     - technical failures: the model predicts the responsible entity from the logs;
       the rule baseline ("blame the hop after the last one that answered") does too.
     - business/policy declines: the response code names its owner (codes.py), which
       is what a receipt alone can tell you.
4. Score both against the hidden ground truth.

Run from the repo root: .venv/bin/python -m examples.run_panel_test
"""

import pandas as pd

from nairatrace import codes, localise, scenarios, sim

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

PANEL = {"people": 8, "per_person": 10, "bank": "mixed"}
PANEL_RUN = {"seed": 8, "overrides": {"simulation.days": 5, "simulation.daily_transactions": 10000}}
TRAIN_RUNS = [
    {"seed": 1, "overrides": {"simulation.days": 4, "simulation.daily_transactions": 20000}},
    {"seed": 2, "overrides": {**scenarios.overrides("stress"), "simulation.days": 3, "simulation.daily_transactions": 20000}},
    {"seed": 3, "overrides": {**scenarios.overrides("stress"), "simulation.days": 3, "simulation.daily_transactions": 20000}},
    # Rare network-side outages (scheme, switch, PTSA, NIBSS) made common, so the model can learn them.
    {"seed": 4, "overrides": {**scenarios.overrides("core"), "simulation.days": 3, "simulation.daily_transactions": 20000}},
]

ROLE_OF_OWNER = {"ISSUER": "ISSUER", "SWITCH": "SWITCH", "ACQUIRER": "ACQUIRER", "PTSA": "TERMINAL",
                 "PRINCIPAL": "PRINCIPAL", "SENDER_FI": "SENDER_FI", "BENEFICIARY_FI": "BENEFICIARY_FI", "NIBSS": "NIBSS"}


def main():
    print("Training on", len(TRAIN_RUNS), "separate simulated worlds (seeds 1-4), never the panel world (seed 8)...")
    train = pd.concat([localise.dataset(sim.run(**cfg)) for cfg in TRAIN_RUNS])
    print(f"  {len(train):,} technical failures to learn from\n")
    model = localise.GBMLocaliser().fit(train)

    panel = sim.run(panel=PANEL, **PANEL_RUN)
    X = localise.features(panel).join(panel["ground_truth"].set_index("trace_id"))
    X = X[X.panel_person.notna()]
    failed = X[X.final_status != "APPROVED"].copy()

    tech = failed[~failed.is_business.astype(bool)]
    business = failed[failed.is_business.astype(bool)]

    proba = model.predict_proba(tech) if len(tech) else pd.DataFrame()
    tech = tech.assign(
        model_pred=proba.idxmax(axis=1), model_p=proba.max(axis=1).round(2),
        rule_pred=tech.apply(localise.rule_last_responding_hop, axis=1))
    # The "code says" view for business declines: the owner of the code on the receipt.
    business = business.assign(code_pred=business.apply(
        lambda r: ROLE_OF_OWNER.get(codes.SPACES[r.code_space][r.final_code].owner), axis=1))

    print(f"Panel: {len(X)} transactions, {len(failed)} not successful "
          f"({len(tech)} technical, {len(business)} business or policy)\n")

    print("TECHNICAL FAILURES: model vs rule vs truth")
    cols = ["panel_person", "issuer", "final_code", "true", "model_pred", "model_p", "rule_pred"]
    t = tech.assign(true=tech.root_role + " / " + tech.fault_type)
    print(t.sort_values("ts")[cols].to_string(index=False))
    print(f"\n  model correct: {(tech.model_pred == tech.root_role).sum()}/{len(tech)}"
          f"   rule correct: {(tech.rule_pred == tech.root_role).sum()}/{len(tech)}\n")

    print("BUSINESS / POLICY DECLINES: what the receipt code says vs truth")
    b = business.assign(true=business.root_role + " / " + business.fault_type)
    print(b.sort_values("ts")[["panel_person", "issuer", "final_code", "true", "code_pred"]].to_string(index=False))
    print(f"\n  code owner correct: {(business.code_pred == business.root_role).sum()}/{len(business)}")

    print("\nExplanations the model gives for each technical failure:")
    for tid, row in tech.sort_values("ts").iterrows():
        print(f"  {row.panel_person} ({row.issuer}, code {row.final_code if pd.notna(row.final_code) else 'none'}),"
              f" truth {row.root_role}/{row.fault_type}\n    {localise.explain(row, proba.loc[tid])}")


if __name__ == "__main__":
    main()
