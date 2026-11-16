import numpy as np
import pandas as pd
import pytest
import simpy

from nairatrace import codes, localise, params, sim
from nairatrace.entities import DOWN, UP, Entity
from nairatrace.world import Calendar

SMALL = {"simulation.days": 1, "simulation.daily_transactions": 4000}


@pytest.fixture(scope="module")
def run():
    return sim.run(seed=7, overrides=SMALL)


def test_every_code_has_one_owner_and_known_kind():
    for space, table in codes.SPACES.items():
        for code, c in table.items():
            assert c.owner and c.kind in ("business", "technical", "none"), (space, code)
            assert c.outcome in ("approved", "declined", "unknown")


def test_unknown_outcome_codes_match_report():
    unknown = {(s, c) for s, t in codes.SPACES.items() for c, v in t.items() if v.outcome == "unknown"}
    assert unknown == {("CARD", "68"), ("CARD", "96"),
                       ("NIP", "09"), ("NIP", "68"), ("NIP", "91"), ("NIP", "96"), ("NIP", "97")}


def test_old_v1_code_misuse_is_gone():
    assert codes.CARD["00"].meaning == "Approved"
    assert codes.CARD["14"].meaning == "Invalid card number"
    assert "11" not in codes.CARD and "19" not in codes.CARD and "28" not in codes.CARD


def test_every_parameter_has_provenance():
    values, provs = params.load()

    def walk(v, p, path=""):
        if isinstance(p, dict):
            for k in p:
                walk(v[k], p[k], f"{path}.{k}")
        else:
            assert isinstance(p, str) and p.strip(), path
    walk(values, provs)
    assert "faults.by_role" in params.estimates(provs)


def test_markov_availability_matches_formula():
    p, _ = params.load()
    cfg = {"mtbf_h": 2, "mttr_median_min": 10, "p_down": 1.0}
    p["faults"]["by_role"]["SWITCH"] = cfg
    p["faults"]["mttr_sigma"] = 0.5
    p["faults"]["peak_fault_multiplier"] = 1.0
    env, episodes = simpy.Environment(), []
    Entity(env, "SWITCH", "X", p, seed=3, episodes=episodes, calendar=Calendar("2026-06-10"))
    horizon = 3000 * 3600
    env.run(until=horizon)
    down = sum(e["t_end"] - e["t_start"] for e in episodes)
    mean_mttr = 600 * np.exp(0.5 ** 2 / 2)
    expected_unavail = mean_mttr / (7200 + mean_mttr)
    assert abs(down / horizon - expected_unavail) < 0.02


def test_cbn_cashout_caps_are_enforced(run):
    t = run["transactions"]
    a = t[(t.route == "A") & (t.final_status == "APPROVED")].copy()
    a["day"] = (a.ts // 86400).astype(int)
    assert a.groupby(["customer_id", "day"]).amount_kobo.sum().max() <= 100_000 * 100
    assert a.groupby(["agent_id", "day"]).amount_kobo.sum().max() <= 1_200_000 * 100


def test_same_seed_is_reproducible():
    a = sim.run(seed=11, overrides=SMALL)["transactions"]
    b = sim.run(seed=11, overrides=SMALL)["transactions"]
    pd.testing.assert_frame_equal(a, b)


def test_debited_failures_are_resolved(run):
    t = run["transactions"]
    debited = t[t.customer_debited]
    assert len(debited) > 0
    assert debited.resolution.isin(["REVERSAL", "LATE_CREDIT", "TSQ_SUCCESS"]).all()
    assert debited.resolved_at.notna().all()


def test_ground_truth_only_for_failures(run):
    m = run["transactions"].merge(run["ground_truth"], on="trace_id")
    assert m[m.final_status == "APPROVED"].root_entity.isna().all()
    assert m[m.final_status.isin(["FAILED", "DECLINED", "DEBITED_FAILED"])].root_entity.notna().all()


def test_observable_tables_hold_no_ground_truth(run):
    hidden = {"root_entity", "root_role", "fault_type", "episode_id", "is_business"}
    assert not hidden & set(run["spans"].columns)
    assert not hidden & set(run["transactions"].columns)
    assert not hidden & set(localise.CAT_COLS + localise.NUM_COLS)


def test_card_codes_come_from_card_table(run):
    s = run["spans"]
    for space, table in codes.SPACES.items():
        used = set(s[(s.code_space == space) & s.code.notna()].code)
        assert used <= set(table), (space, used - set(table))


def test_every_code_has_an_outcome_category():
    from nairatrace import classify
    for space, table in codes.SPACES.items():
        for code in table:
            assert code in classify.CATEGORY[space], (space, code)


def test_classification_columns(run):
    from nairatrace import classify
    t = run["transactions"]
    assert set(t.outcome_category) <= set(classify.CATEGORIES)
    assert (t[t.customer_debited].outcome_category == "Debited but failed").all()
    assert (t[t.final_status == "APPROVED"].outcome_category == "Successful").all()
    assert set(t.part_of_day) <= {"Night", "Morning", "Afternoon", "Evening", "Late evening"}
    assert (t.response_message != "").all()
    assert t.datetime.dt.date.min().isoformat() == run["params"]["simulation"]["start_date"]


def test_card_numbers_are_masked_and_luhn_valid(run):
    from nairatrace import world
    t = run["transactions"]
    cards = t[t.route.isin(["A", "D"])]
    assert cards.masked_pan.str.fullmatch(r"\d{6}\*{6}\d{4}").all()
    assert not t.astype(str).apply(lambda col: col.str.fullmatch(r"\d{16}")).any().any()
    rng = np.random.default_rng(0)
    for _ in range(50):
        pan = world._luhn_pan("506", rng)
        digits = [int(d) for d in pan[::-1]]
        assert sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits)) % 10 == 0


def test_receipt_fields(run):
    from nairatrace import classify
    t = run["transactions"]
    a = t[t.route == "A"]
    assert a.terminal_id.str.fullmatch(r"2[A-Z]{3}[0-9A-F]{4}").all()
    assert a.account_selected.isin(["DEFAULT", "SAVINGS", "CURRENT"]).all()
    assert a[a.final_status == "APPROVED"].auth_code.notna().all()
    assert a[a.final_code.isin(["52", "53"])].account_selected.isin(["SAVINGS", "CURRENT"]).all()
    r = classify.receipt_view(t)
    assert "customer_id" not in r and "path_switch" not in r


def test_auth_code_only_when_approval_reached_terminal(run):
    t = run["transactions"]
    assert t[t.final_status != "APPROVED"].auth_code.isna().all()


def test_panel_same_people_same_bank_different_terminals():
    r = sim.run(seed=5, overrides={"simulation.days": 2, "simulation.daily_transactions": 2000},
                panel={"people": 8, "per_person": 10, "bank": "GTBANK"})
    p = r["transactions"][r["transactions"].panel_person.notna()]
    assert len(p) == 80 and p.panel_person.nunique() == 8
    assert (p.groupby("panel_person").size() == 10).all()
    assert (p.issuer == "GTBANK").all()
    assert (p.groupby("panel_person").terminal_id.nunique() == 10).all()
