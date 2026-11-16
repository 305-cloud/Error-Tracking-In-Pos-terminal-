"""The simulated population: calendar, entities, customers, agents.

Customers hold an account at an issuer (bank or MFB) and usually a card on a
scheme. Agents belong to exactly one principal (Moniepoint, OPay or PalmPay;
CBN exclusivity rule from 1 Apr 2026) and own one terminal that is routed
through one PTSA (NIBSS or UPSL) and the principal's processor/switch.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .entities import Entity, rng_for

DAY = 86400


class Calendar:
    def __init__(self, start_date):
        self.start = date.fromisoformat(start_date)

    def date_at(self, t):
        return self.start + timedelta(seconds=t)

    def is_peak_day(self, t):
        d = self.date_at(t)
        return d.day >= 25 or d.day == 1 or d.month == 12


@dataclass
class Customer:
    cid: str
    issuer: str
    scheme: str
    account_type: str        # SAVINGS | CURRENT
    pan: str                 # full card number (never exported; only the masked form is)
    kyc_tier: int
    balance: float
    expired: bool
    pin_skill: float         # probability of entering the right PIN
    fraud_flag: bool
    fi: str = ""             # NIP institution of the customer's account (same as issuer)
    spent_today: dict = field(default_factory=dict)   # day -> naira cashed out at agents
    spent_week: dict = field(default_factory=dict)    # week -> naira (all channels)
    pin_fails: dict = field(default_factory=dict)     # day -> consecutive wrong PINs


@dataclass
class Agent:
    aid: str
    principal: str
    terminal: str            # 8-character terminal ID
    ptsp: str                # company that deployed and manages the terminal
    default_account: str     # account-type setting left on the terminal: DEFAULT | SAVINGS | CURRENT
    ptsa: str
    processor: str
    cashout_today: dict = field(default_factory=dict)


def _luhn_pan(prefix, rng, length=16):
    """A Luhn-valid card number starting with the scheme prefix."""
    body = prefix + "".join(str(d) for d in rng.integers(0, 10, length - len(prefix) - 1))
    total = 0
    for i, d in enumerate(int(x) for x in reversed(body)):
        total += (d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 == 0 else d
    return body + str((10 - total % 10) % 10)


def mask_pan(pan):
    return pan[:6] + "*" * (len(pan) - 10) + pan[-4:]


def _choice(rng, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


class World:
    def __init__(self, env, params, seed, faults_enabled=True, disabled_faults=()):
        self.env = env
        self.p = params
        self.seed = seed
        self.calendar = Calendar(params["simulation"]["start_date"])
        self.episodes = []
        self.entities = {}
        pop = params["population"]

        def add(role, name):
            enabled = faults_enabled and f"{role}:{name}" not in disabled_faults
            e = Entity(env, role, name, params, seed, self.episodes, self.calendar, enabled)
            self.entities[e.fqn] = e
            return e

        for name in pop["ptsa_share"]:
            add("PTSA", name)
        for name in sorted(set(pop["principal_processor"].values()) | set(pop["gateway_processor"].values())):
            add("SWITCH", name)
        for name in pop["scheme_share"]:
            add("SCHEME", name)
        for name in pop["issuer_share"]:
            add("ISSUER", name)          # card authorisation host of the bank
            add("SENDER_FI", name)       # NIP outward channel of the same bank
            add("BENEFICIARY_FI", name)  # NIP inward channel of the same bank
        for name in pop["principals"]:
            add("PRINCIPAL", name)       # wallet ledger that credits the agent
        for name in pop["gateway_share"]:
            add("GATEWAY", name)
        add("NIBSS", "NIP")

        rng = rng_for(seed, "population")
        self.customers = [self._make_customer(i, rng) for i in range(pop["customers"])]
        self.agents = [self._make_agent(i, rng) for i in range(pop["agents"])]

    def e(self, role, name):
        return self.entities[f"{role}:{name}"]

    def _make_customer(self, i, rng):
        pop = self.p["population"]
        b = pop["balance_naira"]
        issuer = _choice(rng, pop["issuer_share"])
        scheme = _choice(rng, pop["scheme_share"])
        return Customer(
            cid=f"C{i:06d}",
            issuer=issuer,
            fi=issuer,
            scheme=scheme,
            pan=_luhn_pan(pop["card_bin_prefix"][scheme], rng),
            account_type="SAVINGS" if rng.random() < 0.8 else "CURRENT",
            kyc_tier=int(_choice(rng, {int(k): v for k, v in pop["kyc_tier_share"].items()})),
            balance=float(rng.lognormal(np.log(b["median"]), b["sigma"])),
            expired=rng.random() < self.p["issuer_behaviour"]["p_expired_card"],
            pin_skill=float(1 - self.p["issuer_behaviour"]["p_pin_wrong"] * rng.uniform(0.3, 1.7)),
            fraud_flag=rng.random() < self.p["issuer_behaviour"]["p_fraud_block"],
        )

    def _make_agent(self, i, rng):
        pop = self.p["population"]
        principal = _choice(rng, pop["principals"])
        ptsp = pop["principal_ptsp"][principal]
        return Agent(
            aid=f"A{i:05d}",
            principal=principal,
            terminal="2" + ptsp[:3] + "".join(rng.choice(list("0123456789ABCDEF"), 4)),
            ptsp=ptsp,
            default_account=_choice(rng, pop["terminal_default_account"]),
            ptsa=_choice(rng, pop["ptsa_share"]),
            processor=pop["principal_processor"][principal],
        )

    def day_index(self, t):
        return int(t // DAY)

    def week_index(self, t):
        return int(t // (7 * DAY))
