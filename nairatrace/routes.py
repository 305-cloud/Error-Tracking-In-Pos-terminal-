"""The four Nigerian payment routes as SimPy processes.

A  card cash-out at an agent POS
     Terminal -> PTSA[NIBSS|UPSL] -> Switch -> Scheme -> Issuer ; then Principal credits the agent wallet
B  transfer-funded cash-out
     Customer app -> Sender FI -> NIBSS NIP -> Beneficiary FI (agent's principal MFB) ; then wallet alert
C  NIP transfer
     Customer app -> Sender FI -> [on-us shortcut] -> NIBSS NIP (NE, then FT, TSQ if unknown) -> Beneficiary FI
D  online card checkout
     Checkout -> Gateway[Paystack|Flutterwave] -> Switch -> Scheme -> Issuer (3DS/OTP)

Every hop is a SimPy process that calls the next one and waits for it with its
own timeout. A hop that hears nothing answers upstream itself (card 68, NIP
97/68), while the downstream work carries on in the background - which is how
a customer ends up debited for a transaction that "failed". Card reversals
(MTI 0420) and NIP status queries (TSQ) then resolve the money later.

Ground truth is set where the failure is caused and is never written to spans.
"""

from dataclasses import dataclass, field

import numpy as np

from . import codes
from .entities import DOWN, DEGRADED, rng_for
from .world import mask_pan


@dataclass
class Result:
    code: str | None          # None = no response code at all (terminal/app saw nothing)
    space: str                # CARD | NIP | LOCAL
    status: str               # span status: OK | DECLINE | TIMEOUT | ERROR | NO_RESPONSE
    root: str | None = None   # ground truth: entity fqn that caused it
    root_role: str | None = None
    fault: str | None = None  # DOWN | DEGRADED | OVERLOAD | TERMINAL_NETWORK | POLICY:* | BUSINESS:* | WEBHOOK_LOST
    episode: str | None = None

    @property
    def approved(self):
        return self.code == "00"


@dataclass
class Transaction:
    trace_id: str
    route: str
    customer: object
    agent: object
    amount: float
    ts: float
    retry_no: int = 0
    retry_of: str | None = None
    stan: str = ""
    rrn: str = ""
    path: dict = field(default_factory=dict)      # observable routing: ptsa, switch, scheme, gateway, fi...
    excess: dict = field(default_factory=dict)    # fqn -> (excess latency s, state, episode) [hidden]
    debited: bool = False
    credited: bool = False                        # NIP: beneficiary credited
    closed: bool = False                          # the customer has been shown a final answer
    final_status: str = "IN_FLIGHT"
    final_code: str | None = None
    final_space: str | None = None
    customer_debited: bool = False                # debited while being shown a failure
    resolution: str | None = None
    resolved_at: float | None = None
    t_end: float | None = None
    truth: Result | None = None
    panel: bool = False                           # part of a fixed study panel (no automatic retries)
    bene_draw: float | None = None                # beneficiary account condition, fixed across NE/FT
    account_selected: str | None = None           # card: DEFAULT | SAVINGS | CURRENT
    auth_code: str | None = None                  # card: issuer authorisation code on approval

    def observable_row(self):
        card = self.route in ("A", "D")
        return {
            "trace_id": self.trace_id, "route": self.route, "customer_id": self.customer.cid,
            "panel_person": getattr(self.customer, "panel_label", None) if self.panel else None,
            "agent_id": self.agent.aid if self.agent else None, "issuer": self.customer.issuer,
            "terminal_id": self.agent.terminal if self.route == "A" else None,
            "ptsp": self.agent.ptsp if self.route == "A" else None,
            "masked_pan": mask_pan(self.customer.pan) if card else None,
            "card_bin": self.customer.pan[:6] if card else None,
            "account_selected": self.account_selected,
            "stan": self.stan, "rrn": self.rrn,
            # The terminal only prints an auth code when the approval actually reached it.
            "auth_code": self.auth_code if self.final_status == "APPROVED" else None,
            **{f"path_{k}": v for k, v in self.path.items()},
            "amount_kobo": int(round(self.amount * 100)), "ts": self.ts, "t_end": self.t_end,
            "retry_no": self.retry_no, "retry_of": self.retry_of,
            "final_status": self.final_status, "final_code": self.final_code, "code_space": self.final_space,
            "customer_debited": self.customer_debited, "resolution": self.resolution,
            "resolved_at": self.resolved_at,
        }

    def truth_row(self):
        t = self.truth
        failed = self.final_status not in ("APPROVED",)
        return {
            "trace_id": self.trace_id,
            "root_entity": t.root if (t and failed) else None,
            "root_role": t.root_role if (t and failed) else None,
            "fault_type": t.fault if (t and failed) else None,
            "episode_id": t.episode if (t and failed) else None,
            "is_business": bool(t and failed and t.fault and t.fault.startswith(("BUSINESS", "POLICY"))),
        }


# Code a caller emits when the next hop refuses the connection or times out.
DOWN_CODE = {("CARD", "PTSA"): "91", ("CARD", "SWITCH"): "91", ("CARD", "SCHEME"): "91",
             ("CARD", "GATEWAY"): "91",
             ("NIP", "SENDER_FI"): "96", ("NIP", "NIBSS"): "91"}
TIMEOUT_CODE = {("CARD", "PTSA"): "68", ("CARD", "SWITCH"): "68", ("CARD", "SCHEME"): "68",
                ("CARD", "GATEWAY"): "68",
                ("NIP", "SENDER_FI"): "68", ("NIP", "NIBSS"): "97"}
RETRYABLE_BUSINESS = {"55", "05", "52", "53", "L10"}


class Router:
    def __init__(self, env, world, tracer, params, seed):
        self.env, self.w, self.tr, self.p, self.seed = env, world, tracer, params, seed
        self.timeouts = params["timing"]["timeouts_s"]
        self.beh = params["issuer_behaviour"]
        self.reg = params["regulation"]
        self.f = params["faults"]
        self.rng = rng_for(seed, "router")

    # ------------------------------------------------------------------ helpers
    def _r(self, ent, code, space, status, fault, episode=None):
        return Result(code, space, status, ent.fqn, ent.role, fault, episode)

    def _slowest(self, tx, hops):
        """Ground truth for a timeout: the downstream hop with the largest excess latency."""
        seen = [(tx.excess[h.fqn], h) for h in hops if h.fqn in tx.excess]
        if not seen:
            h = hops[0]
            return h, ("DEGRADED" if h.state == DEGRADED else "OVERLOAD"), h.episode_id
        (excess, state, episode), h = max(seen, key=lambda x: x[0][0])
        return h, ("DEGRADED" if state == DEGRADED else "OVERLOAD"), episode

    def _lognormal(self, spec, rng):
        return float(rng.lognormal(np.log(spec["median"]), spec["sigma"]))

    # ------------------------------------------------------------- generic hop
    def _hop(self, tx, hops, i, op, space, parent_id, rng):
        env, ent = self.env, hops[i]
        span_id = self.tr.new_span_id()
        t0 = env.now
        lat, nominal = ent.sample_latency_s()
        tx.excess[ent.fqn] = (lat - nominal, ent.state, ent.episode_id)
        ent.inflight += 1
        try:
            yield env.timeout(lat / 2)
            res = self._decide(tx, ent, op, space, rng)
            if res is None and ent.degraded_error():
                res = self._r(ent, "96", space, "ERROR", "DEGRADED", ent.episode_id)
            if res is None and i + 1 < len(hops):
                nxt = hops[i + 1]
                if nxt.state == DOWN:
                    yield env.timeout(0.02)   # connection refused: fast failure
                    res = Result(DOWN_CODE[(space, ent.role)], space, "ERROR",
                                 nxt.fqn, nxt.role, "DOWN", nxt.episode_id)
                else:
                    proc = env.process(self._hop(tx, hops, i + 1, op, space, span_id, rng))
                    yield proc | env.timeout(self.timeouts[ent.role])
                    if proc.triggered:
                        res = proc.value
                    else:
                        h, fault, ep = self._slowest(tx, hops[i + 1:])
                        res = Result(TIMEOUT_CODE[(space, ent.role)], space, "TIMEOUT", h.fqn, h.role, fault, ep)
            if res is None:
                res = self._final(tx, ent, op, space)
            yield env.timeout(lat / 2)
        finally:
            ent.inflight -= 1
        mti = {"AUTH": "0200", "NE": "NE", "FT": "FT"}[op]
        self.tr.span(tx, span_id, parent_id, ent.fqn, ent.role, op, mti, t0, env.now,
                     res.status, res.code, res.space)
        return res

    def _outer(self, tx, hops, op, space, caller_role, parent_id, rng):
        """The terminal/app/checkout waiting on the first hop."""
        env = self.env
        first = hops[0]
        if first.state == DOWN:
            yield env.timeout(float(rng.uniform(1, 10)))   # host unreachable
            return Result(None, space, "NO_RESPONSE", first.fqn, first.role, "DOWN", first.episode_id)
        proc = env.process(self._hop(tx, hops, 0, op, space, parent_id, rng))
        yield proc | env.timeout(self.timeouts[caller_role])
        if proc.triggered:
            return proc.value
        h, fault, ep = self._slowest(tx, hops)
        # The work may still finish later; _after_debit() reverses it if so.
        return Result(None, space, "NO_RESPONSE", h.fqn, h.role, fault, ep)

    # ------------------------------------------------------- entity decisions
    def _decide(self, tx, ent, op, space, rng):
        """Checks an entity makes before forwarding. None = pass."""
        b, c = self.beh, tx.customer
        if ent.role == "PTSA" and rng.random() < self.reg["terminal_outside_fence_share"]:
            # Terminal moved off its registered site: geo-fence rejects (CBN 70 m radius).
            return Result("58", space, "DECLINE", f"TERMINAL:{tx.agent.terminal}", "TERMINAL", "POLICY:GEOFENCE")
        if ent.role == "SWITCH" and tx.retry_no > 0 and rng.random() < self.beh["p_duplicate_on_retry"]:
            # The retry reaches the switch looking like the earlier attempt: rejected as a duplicate.
            return self._r(ent, "94", space, "ERROR", "DUPLICATE")
        if ent.role == "ISSUER":
            return self._issuer_checks(tx, ent, space, rng)
        if ent.role == "SENDER_FI" and op == "FT":
            day, week = self.w.day_index(self.env.now), self.w.week_index(self.env.now)
            if tx.amount > self.reg["kyc_tier_txn_limit"][c.kyc_tier]:
                return self._r(ent, "61", space, "DECLINE", "BUSINESS:61")
            if tx.route == "B" and c.spent_week.get(week, 0) + tx.amount > self.reg["cashout_cap_customer_week"]:
                return Result("L03", "LOCAL", "DECLINE", ent.fqn, ent.role, "POLICY:CAP_WEEK")
            if c.balance < tx.amount:
                return self._r(ent, "51", space, "DECLINE", "BUSINESS:51")
            if rng.random() < b["nip_p_not_permitted"]:
                return self._r(ent, "57", space, "DECLINE", "BUSINESS:57")
            # Debit (hold) before forwarding: from here on a failure leaves the customer debited.
            c.balance -= tx.amount
            tx.debited = True
            if tx.route == "B":
                c.spent_week[week] = c.spent_week.get(week, 0) + tx.amount
            return None
        if ent.role == "BENEFICIARY_FI" and op == "NE":
            if tx.bene_draw is None:
                tx.bene_draw = rng.random()
            u = tx.bene_draw
            if u < b["nip_p_invalid_account"]:
                return self._r(ent, "07", space, "DECLINE", "BUSINESS:07")
            if u < b["nip_p_invalid_account"] + b["nip_p_dormant"]:
                return self._r(ent, "06", space, "DECLINE", "BUSINESS:06")
            if u < b["nip_p_invalid_account"] + b["nip_p_dormant"] + b["nip_p_fraud"]:
                return self._r(ent, "34", space, "DECLINE", "BUSINESS:34")
            if u < b["nip_p_invalid_account"] + b["nip_p_dormant"] + b["nip_p_fraud"] + b["nip_p_name_mismatch"]:
                return self._r(ent, "08", space, "DECLINE", "BUSINESS:08")
        return None

    def _issuer_checks(self, tx, ent, space, rng):
        b, c, now = self.beh, tx.customer, self.env.now
        day, week = self.w.day_index(now), self.w.week_index(now)

        def decline(code, space_=space, fault=None):
            return Result(code, space_, "DECLINE", ent.fqn, ent.role, fault or f"BUSINESS:{code}")

        if c.expired:
            return decline("54")
        if rng.random() < b["p_invalid_card"]:
            return decline("14")
        if tx.route == "A":
            if c.pin_fails.get(day, 0) >= 3:
                return decline("75")
            if rng.random() > c.pin_skill:
                c.pin_fails[day] = c.pin_fails.get(day, 0) + 1
                return decline("55")
            c.pin_fails[day] = 0
            # A wrong account type (often the terminal's default setting) is declined: NIBSS-named cause.
            sel = tx.account_selected
            if sel == "CURRENT" and c.account_type != "CURRENT":
                return decline("52")
            if sel == "SAVINGS" and c.account_type != "SAVINGS":
                return decline("53")
        if tx.route == "D" and rng.random() < b["p_otp_fail"]:
            return decline("L10", "LOCAL")
        if c.fraud_flag:
            return decline("59")
        if rng.random() < b["p_not_permitted"] * (3 if tx.route == "D" else 1):
            return decline("57")
        if tx.route == "A" and c.spent_week.get(week, 0) + tx.amount > self.reg["cashout_cap_customer_week"]:
            return decline("L03", "LOCAL", "POLICY:CAP_WEEK")
        if tx.amount > self.reg["kyc_tier_txn_limit"][c.kyc_tier]:
            return decline("61")
        if c.balance < tx.amount:
            return decline("51")
        if rng.random() < b["p_do_not_honour"]:
            return decline("05")
        return None

    def _final(self, tx, ent, op, space):
        """The last hop accepts: issuer debits, beneficiary FI credits."""
        now = self.env.now
        if ent.role == "ISSUER":
            c = tx.customer
            c.balance -= tx.amount
            tx.debited = True
            tx.auth_code = f"{int(rng_for(self.seed, f'auth:{tx.trace_id}').integers(0, 1_000_000)):06d}"
            if tx.route == "A":
                week = self.w.week_index(now)
                c.spent_week[week] = c.spent_week.get(week, 0) + tx.amount
            self._after_debit(tx)
        elif ent.role == "BENEFICIARY_FI" and op == "FT":
            tx.credited = True
        return Result("00", space, "OK", ent.fqn, ent.role, None)

    # ---------------------------------------------------- money resolution
    def _after_debit(self, tx):
        """Card: if the customer was already shown a failure, the debit must be reversed (0420)."""
        if tx.debited and tx.closed and tx.final_status not in ("APPROVED",) and tx.resolution is None:
            tx.customer_debited = True
            tx.final_status = "DEBITED_FAILED"
            tx.resolution = "PENDING_REVERSAL"
            self.env.process(self._reverse(tx))

    def _reverse(self, tx, kind="REVERSAL"):
        rng = rng_for(self.seed, f"reversal:{tx.trace_id}")
        yield self.env.timeout(self._lognormal(self.p["timing"]["reversal_delay_s"], rng))
        if kind == "REVERSAL":
            tx.customer.balance += tx.amount
        tx.resolution, tx.resolved_at = kind, self.env.now

    def _tsq(self, tx):
        """NIP: query the status of an unknown-outcome transfer until it resolves."""
        rng = rng_for(self.seed, f"tsq:{tx.trace_id}")
        for _ in range(6):
            yield self.env.timeout(self._lognormal(self.p["timing"]["tsq_delay_s"], rng))
            if tx.credited:
                tx.final_status, tx.final_code = "APPROVED_LATE", "00"
                tx.resolution, tx.resolved_at = "TSQ_SUCCESS", self.env.now
                return
        tx.final_status = "DEBITED_FAILED"
        tx.customer_debited = True
        yield from self._reverse(tx)

    def _principal_credit(self, tx, principal, parent):
        """Principal credits the agent wallet; logged as a span so the step is observable."""
        span_id, t0 = self.tr.new_span_id(), self.env.now
        lat, _ = principal.sample_latency_s()
        yield self.env.timeout(lat)
        self.tr.span(tx, span_id, parent, principal.fqn, "PRINCIPAL", "CREDIT", "-", t0, self.env.now,
                     "OK", "00", "LOCAL")

    # ----------------------------------------------------------------- routes
    def start(self, route, customer, agent, amount, retry_no=0, retry_of=None, panel=False):
        tx = Transaction(self.tr.new_trace_id(), route, customer, agent, amount, self.env.now,
                         retry_no, retry_of)
        tx.panel = panel
        tx.stan = f"{int(tx.trace_id[2:]) % 1000000:06d}"
        tx.rrn = f"{self.w.calendar.date_at(self.env.now):%y%m%d}{tx.stan}"
        proc = {"A": self.route_a, "B": self.route_b, "C": self.route_c, "D": self.route_d}[route]
        self.env.process(self._run(tx, proc))

    def _run(self, tx, proc):
        rng = rng_for(self.seed, f"tx:{tx.trace_id}")
        span_id = self.tr.new_span_id()
        t0 = self.env.now
        res = yield from proc(tx, span_id, rng)
        self._close(tx, res)
        root_role = {"A": "TERMINAL", "B": "APP", "C": "APP", "D": "CHECKOUT"}[tx.route]
        name = tx.agent.terminal if (tx.route == "A") else tx.customer.cid
        self.tr.span(tx, span_id, None, f"{root_role}:{name}", root_role, "REQUEST", "-", t0, self.env.now,
                     res.status, res.code, res.space)
        self.tr.finish(tx)
        # A customer may retry after a technical failure or a correctable decline.
        if not res.approved and not tx.panel and tx.retry_no < self.p["demand"]["max_retries"]:
            technical = res.fault and not res.fault.startswith(("BUSINESS", "POLICY"))
            if (technical or res.code in RETRYABLE_BUSINESS) and rng.random() < self.p["demand"]["retry_probability"]:
                yield self.env.timeout(float(rng.uniform(20, 120)))
                self.start(tx.route, tx.customer, tx.agent, tx.amount, tx.retry_no + 1, tx.trace_id)

    def _close(self, tx, res):
        tx.closed, tx.t_end = True, self.env.now
        tx.final_code, tx.final_space = res.code, res.space
        if res.approved:
            if tx.final_status == "IN_FLIGHT":
                tx.final_status = "APPROVED"
            # APPROVED_LATE keeps the cause of the delay recorded before the TSQ.
            tx.truth = tx.truth if tx.final_status != "APPROVED" else None
            return
        tx.truth = res
        if tx.final_status == "IN_FLIGHT":
            business = res.fault and res.fault.startswith(("BUSINESS", "POLICY"))
            tx.final_status = "DECLINED" if business else "FAILED"
        if tx.route in ("A", "D"):
            self._after_debit(tx)

    def route_a(self, tx, parent, rng):
        w, a, c, env = self.w, tx.agent, tx.customer, self.env
        tx.path = {"ptsa": a.ptsa, "switch": a.processor, "scheme": c.scheme, "principal": a.principal}
        yield env.timeout(float(rng.lognormal(np.log(1.5), 0.4)))   # card read + PIN entry
        # Account type: the terminal's setting unless the customer picks one.
        if rng.random() < self.beh["p_customer_picks_account"]:
            wrong = rng.random() < self.beh["p_customer_picks_wrong"]
            other = "CURRENT" if c.account_type == "SAVINGS" else "SAVINGS"
            tx.account_selected = other if wrong else c.account_type
        else:
            tx.account_selected = a.default_account
        day = w.day_index(env.now)
        principal = w.e("PRINCIPAL", a.principal)
        if c.spent_today.get(day, 0) + tx.amount > self.reg["cashout_cap_customer_day"]:
            return Result("L01", "LOCAL", "DECLINE", principal.fqn, "PRINCIPAL", "POLICY:CAP_CUSTOMER_DAY")
        if a.cashout_today.get(day, 0) + tx.amount > self.reg["cashout_cap_agent_day"]:
            return Result("L02", "LOCAL", "DECLINE", principal.fqn, "PRINCIPAL", "POLICY:CAP_AGENT_DAY")
        terminal = f"TERMINAL:{a.terminal}"
        lost = rng.random() < self.f["terminal_network_loss_p"]
        lost_after = lost and rng.random() < self.f["terminal_loss_after_debit_share"]
        if lost and not lost_after:
            # No signal: the terminal either fails to connect quickly or hangs until its timeout.
            wait = float(rng.uniform(1, 10)) if rng.random() < 0.5 else self.timeouts["TERMINAL"]
            yield env.timeout(wait)
            return Result(None, "CARD", "NO_RESPONSE", terminal, "TERMINAL", "TERMINAL_NETWORK")
        hops = [w.e("PTSA", a.ptsa), w.e("SWITCH", a.processor), w.e("SCHEME", c.scheme), w.e("ISSUER", c.issuer)]
        res = yield from self._outer(tx, hops, "AUTH", "CARD", "TERMINAL", parent, rng)
        if res.approved and lost_after:
            # Issuer debited, but the response never reached the terminal: auto-reversal follows.
            return Result(None, "CARD", "NO_RESPONSE", terminal, "TERMINAL", "TERMINAL_NETWORK")
        if res.approved:
            if principal.state == DOWN:
                # Customer debited, agent wallet not credited, so no cash is handed over.
                return Result(None, "LOCAL", "ERROR", principal.fqn, "PRINCIPAL", "DOWN", principal.episode_id)
            yield from self._principal_credit(tx, principal, parent)
            c.spent_today[day] = c.spent_today.get(day, 0) + tx.amount
            a.cashout_today[day] = a.cashout_today.get(day, 0) + tx.amount
        return res

    def _nip(self, tx, sender_name, bene_name, parent, rng):
        w, env = self.w, self.env
        sender = w.e("SENDER_FI", sender_name)
        tx.path.update({"sender_fi": sender_name, "beneficiary_fi": bene_name})
        if sender.state == DOWN:
            yield env.timeout(3.0)
            return Result(None, "NIP", "NO_RESPONSE", sender.fqn, sender.role, "DOWN", sender.episode_id)
        if sender_name == bene_name:
            # On-us: never reaches NIBSS.
            tx.path["on_us"] = True
            res = yield from self._outer(tx, [sender], "FT", "NIP", "APP", parent, rng)
            if res.approved:
                tx.credited = True
            return res
        hops = [sender, w.e("NIBSS", "NIP"), w.e("BENEFICIARY_FI", bene_name)]
        ne = yield from self._outer(tx, hops, "NE", "NIP", "APP", parent, rng)
        if not ne.approved:
            return ne
        tx.excess.clear()
        ft = yield from self._outer(tx, hops, "FT", "NIP", "APP", parent, rng)
        if ft.approved or not tx.debited:
            return ft
        # Debited but no clean answer: unknown outcome, so query status (TSQ) before refunding.
        tx.final_status = "PENDING"
        tx.truth = ft
        yield from self._tsq(tx)
        return Result("00", "NIP", "OK") if tx.credited else ft

    def route_b(self, tx, parent, rng):
        a, c = tx.agent, tx.customer
        bene = self.p["population"]["principal_fi"][a.principal]
        tx.path["principal"] = a.principal
        res = yield from self._nip(tx, c.fi, bene, parent, rng)
        if res.approved:
            principal = self.w.e("PRINCIPAL", a.principal)
            if principal.state == DOWN:
                # Money reached the agent's principal but the wallet alert did not: no cash handed over.
                tx.customer_debited = True
                tx.final_status = "DEBITED_FAILED"
                tx.resolution = "PENDING_LATE_CREDIT"
                self.env.process(self._reverse(tx, kind="LATE_CREDIT"))
                return Result(None, "LOCAL", "ERROR", principal.fqn, "PRINCIPAL", "DOWN", principal.episode_id)
            yield from self._principal_credit(tx, principal, parent)
        return res

    def route_c(self, tx, parent, rng):
        c = tx.customer
        if rng.random() < self.p["demand"]["on_us_share"]:
            bene = c.fi
        else:
            pop = self.p["population"]["issuer_share"]
            names = [n for n in pop if n != c.fi]
            bene = names[rng.integers(len(names))]
        return (yield from self._nip(tx, c.fi, bene, parent, rng))

    def route_d(self, tx, parent, rng):
        w, c, env = self.w, tx.customer, self.env
        pop = self.p["population"]
        gw_names = list(pop["gateway_share"])
        gw = gw_names[rng.choice(len(gw_names), p=np.array(list(pop["gateway_share"].values())))]
        switch = pop["gateway_processor"][gw]
        tx.path = {"gateway": gw, "switch": switch, "scheme": c.scheme}
        hops = [w.e("GATEWAY", gw), w.e("SWITCH", switch), w.e("SCHEME", c.scheme), w.e("ISSUER", c.issuer)]
        res = yield from self._outer(tx, hops, "AUTH", "CARD", "CHECKOUT", parent, rng)
        if res.approved and rng.random() < self.f["lost_webhook_p"]:
            # Debited, but the merchant never got the webhook: no value delivered.
            gateway = hops[0]
            tx.customer_debited = True
            tx.final_status = "DEBITED_FAILED"
            tx.resolution = "PENDING_REFUND"
            env.process(self._reverse(tx))
            return Result(None, "CARD", "ERROR", gateway.fqn, "GATEWAY", "WEBHOOK_LOST")
        return res
