"""Run a simulation and write its tables.

Output directory layout:
  observable/spans.parquet, observable/transactions.parquet
  hidden/ground_truth.parquet, hidden/episodes.parquet
  provenance.json   (every parameter's source; [E] estimates listed separately)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from . import classify
from . import params as P
from .entities import rng_for
from .routes import Router
from .trace import Tracer
from .world import DAY, World


def _lognormal_amount(spec, rng):
    return float(min(spec["max"], max(100, rng.lognormal(np.log(spec["median"]), spec["sigma"]))))


def arrivals(env, world, router, p, seed):
    """Non-homogeneous Poisson arrivals: piecewise-constant rate per hour of day."""
    d = p["demand"]
    rng = rng_for(seed, "arrivals")
    weights = np.array(d["hourly_weights"], dtype=float)
    weights /= weights.sum()
    routes = list(d["route_mix"])
    mix = np.array([d["route_mix"][r] for r in routes], dtype=float)
    mix /= mix.sum()
    ticket = {"A": d["card_ticket_naira"], "B": d["card_ticket_naira"],
              "C": d["nip_ticket_naira"], "D": d["online_ticket_naira"]}
    for day in range(p["simulation"]["days"]):
        mult = d["month_end_volume_multiplier"] if world.calendar.is_peak_day(day * DAY) else 1.0
        for hour in range(24):
            start = day * DAY + hour * 3600
            n = rng.poisson(p["simulation"]["daily_transactions"] * weights[hour] * mult)
            for t in np.sort(rng.uniform(start, start + 3600, n)):
                yield env.timeout(max(0.0, t - env.now))
                route = routes[rng.choice(len(routes), p=mix)][0]
                customer = world.customers[rng.integers(len(world.customers))]
                agent = world.agents[rng.integers(len(world.agents))] if route in "AB" else None
                amount = _intended_amount(customer, route, _lognormal_amount(ticket[route], rng), p, rng)
                if amount is not None:
                    router.start(route, customer, agent, amount)


def _intended_amount(customer, route, amount, p, rng):
    """Customers usually stay within what they have and what their KYC tier allows."""
    d = p["demand"]
    limit = p["regulation"]["kyc_tier_txn_limit"][customer.kyc_tier]
    if amount > limit and rng.random() >= d["p_over_tier_limit"]:
        amount = limit * float(rng.uniform(0.3, 0.95))
    if amount > customer.balance and rng.random() >= d["p_overdraw"][route]:
        if customer.balance < 500:
            return None   # nearly empty account: the customer does not try
        amount = customer.balance * float(rng.uniform(0.2, 0.9))
    return max(100.0, round(amount, -1))


def income(env, world, p, seed):
    """Inflows so balances do not drain to zero: more people are paid on peak days."""
    d = p["demand"]
    rng = rng_for(seed, "income")
    spec = d["income_naira"]
    for day in range(p["simulation"]["days"]):
        yield env.timeout(max(0.0, day * DAY + 6 * 3600 - env.now))
        prob = d["salary_day_income_p"] if world.calendar.is_peak_day(env.now) else d["daily_income_p"]
        for c in world.customers:
            if rng.random() < prob:
                c.balance += float(rng.lognormal(np.log(spec["median"]), spec["sigma"]))


def panel_process(env, world, router, p, seed, people=8, per_person=10, bank="GTBANK"):
    """A fixed study panel: the same people, all at one bank, each cashing out at a
    different agent terminal every time. Runs alongside normal traffic, so outages and
    busy periods hit the panel as they would anyone else. Panel attempts are not retried
    automatically, so each person has exactly `per_person` transactions."""
    rng = rng_for(seed, "panel")
    horizon = p["simulation"]["days"] * DAY
    picks = rng.choice(len(world.customers), people, replace=False)
    schedule = []
    for n, idx in enumerate(picks, 1):
        c = world.customers[idx]
        c.issuer = c.fi = bank
        c.panel_label = f"Person {n}"
        # Panel members make 10 cash-outs, so they need a working balance, not a near-empty account [E].
        c.balance = max(c.balance, float(rng.lognormal(np.log(120_000), 0.6)))
        agents = rng.choice(len(world.agents), per_person, replace=False)
        # Daytime-weighted times, like the rest of the demand.
        w = np.array(p["demand"]["hourly_weights"], dtype=float); w /= w.sum()
        for a in agents:
            day, hour = rng.integers(p["simulation"]["days"]), rng.choice(24, p=w)
            schedule.append((day * DAY + hour * 3600 + rng.uniform(0, 3600), c, world.agents[a]))
    for t, c, a in sorted(schedule, key=lambda x: x[0]):
        yield env.timeout(max(0.0, t - env.now))
        amount = _intended_amount(c, "A", _lognormal_amount(p["demand"]["card_ticket_naira"], rng), p, rng)
        if amount is None:   # account nearly empty: they try a small amount anyway, which may be declined
            amount = float(max(500, round(c.balance, -2) or 500))
        router.start("A", c, a, amount, panel=True)


def run(seed=1, overrides=None, params_path=P.DEFAULT_PATH, faults_enabled=True, disabled_faults=(), panel=None):
    p, prov = P.load(params_path, overrides)
    env = simpy.Environment()
    world = World(env, p, seed, faults_enabled, disabled_faults)
    tracer = Tracer()
    router = Router(env, world, tracer, p, seed)
    env.process(arrivals(env, world, router, p, seed))
    env.process(income(env, world, p, seed))
    if panel:
        env.process(panel_process(env, world, router, p, seed, **panel))
    horizon = p["simulation"]["days"] * DAY
    env.run(until=horizon + 3 * DAY)   # drain: let reversals and TSQs finish
    for e in world.entities.values():
        e.close_episode()
    spans = pd.DataFrame(tracer.spans)
    txs = classify.add_columns(pd.DataFrame(tracer.transaction_rows()), p["simulation"]["start_date"])
    truth = pd.DataFrame(tracer.ground_truth_rows())
    episodes = pd.DataFrame(world.episodes, columns=["episode_id", "entity", "role", "fault_type", "t_start", "t_end"])
    # Only transactions that started inside the horizon (the drain period has no arrivals anyway).
    return {"spans": spans, "transactions": txs, "ground_truth": truth, "episodes": episodes,
            "params": p, "provenance": prov, "seed": seed}


def write(result, out_dir):
    out = Path(out_dir)
    (out / "observable").mkdir(parents=True, exist_ok=True)
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    result["spans"].to_parquet(out / "observable" / "spans.parquet", index=False)
    result["transactions"].to_parquet(out / "observable" / "transactions.parquet", index=False)
    classify.receipt_view(result["transactions"]).to_csv(out / "observable" / "receipts.csv", index=False)
    result["ground_truth"].to_parquet(out / "hidden" / "ground_truth.parquet", index=False)
    result["episodes"].to_parquet(out / "hidden" / "episodes.parquet", index=False)
    with open(out / "provenance.json", "w") as f:
        json.dump({"seed": result["seed"], "provenance": result["provenance"],
                   "estimates": P.estimates(result["provenance"])}, f, indent=2)
    return out
