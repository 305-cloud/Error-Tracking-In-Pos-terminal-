"""Command line: python -m remsimer2 {simulate,validate,localise} [options]."""

import argparse

import pandas as pd

from . import localise, scenarios, sim, validate

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _overrides(args):
    o = scenarios.overrides(args.scenario)
    if args.days:
        o["simulation.days"] = args.days
    if args.volume:
        o["simulation.daily_transactions"] = args.volume
    return o


def cmd_simulate(args):
    r = sim.run(seed=args.seed, overrides=_overrides(args))
    out = sim.write(r, args.out)
    print(f"{len(r['transactions']):,} transactions, {len(r['spans']):,} spans, "
          f"{len(r['episodes'])} fault episodes -> {out}")


def cmd_validate(args):
    r = sim.run(seed=args.seed, overrides=_overrides(args))
    print("Stylised facts vs public aggregates\n")
    print(validate.stylised_facts(r).to_string(index=False))
    print("\nFault episodes by role\n")
    print(validate.availability(r).to_string(index=False))
    print("\nMost common failure codes\n")
    print(validate.code_distribution(r).to_string(index=False))


def cmd_localise(args):
    print(f"Training on seed {args.seed}, testing on seed {args.seed + 1} (an unseen world and fault history)")
    train = localise.dataset(sim.run(seed=args.seed, overrides=_overrides(args)))
    test = localise.dataset(sim.run(seed=args.seed + 1, overrides=_overrides(args)))
    print(f"technical failures: train {len(train):,}, test {len(test):,}\n")
    print("Root role mix (test):\n" + test.root_role.value_counts().to_string() + "\n")
    rule = test.apply(localise.rule_last_responding_hop, axis=1)
    gbm = localise.GBMLocaliser().fit(train)
    proba = gbm.predict_proba(test)
    print(localise.evaluate(test, proba, rule).to_string() + "\n")
    pred = proba.idxmax(axis=1)
    print("AC@1 by injected fault type (gbm vs rule):")
    by_rule = localise.accuracy_by_fault(test, rule).rename(columns={"AC1": "AC1_rule"}).drop(columns="n")
    print(localise.accuracy_by_fault(test, pred).join(by_rule).to_string() + "\n")
    print("Confusion matrix (gbm, rows = true root role):\n" + localise.confusion(test, pred).to_string() + "\n")
    print("Example verdicts:")
    for tid in test.sample(min(5, len(test)), random_state=0).index:
        row = test.loc[tid]
        print(f"  {tid} route {row.route} truth={row.root_role}/{row.fault_type}\n    "
              + localise.explain(row, proba.loc[tid]))


def main():
    ap = argparse.ArgumentParser(prog="remsimer2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("simulate", cmd_simulate), ("validate", cmd_validate), ("localise", cmd_localise)):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--days", type=int)
        sp.add_argument("--volume", type=int, help="transactions per day")
        sp.add_argument("--scenario", default="default", choices=["default", "stress"])
        if name == "simulate":
            sp.add_argument("--out", default="runs/latest")
        sp.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
