# REMSIMER v2: Nigerian payment failure simulator

A discrete-event simulator (SimPy) of four Nigerian payment routes, with injected faults, hidden ground truth and a fault localiser. It replaces v1's AND-gate rules and its unused MLPs trained on US data. The design follows the research report *REMSIMER Nigerian payment simulation redesign*.

## What it simulates

| Route | Chain |
|---|---|
| A. Card cash-out at an agent POS | Terminal → PTSA (NIBSS or UPSL) → switch (Moniepoint, Interswitch or UP) → scheme (Verve, Mastercard, Visa or AfriGo) → issuer → principal credits the agent wallet |
| B. Transfer-funded cash-out | App → sender bank → NIBSS NIP → agent's principal MFB → wallet alert |
| C. NIP transfer | App → sender bank → NIBSS NIP (name enquiry, transfer, then TSQ if the outcome is unknown) → beneficiary bank. Same-bank transfers skip NIBSS |
| D. Online card checkout | Checkout → Paystack or Flutterwave → switch → scheme → issuer (3DS/OTP) → merchant webhook |

**Entity behaviour.** Each entity (PTSA, switch, scheme, bank channel, principal, gateway, NIBSS) moves between UP, DEGRADED and DOWN, with:
- time to failure drawn from Exp(MTBF)
- lognormal repair times
- faults more likely on month-end and in December
- latency that rises as the entity gets busier

**Timeouts.** Every hop waits for the next one with its own timeout. When nothing comes back, the hop answers upstream itself with code 68 (card) or 97 (NIP), while the work carries on downstream. That is how "debited but failed" happens. It is resolved by a card reversal (0420), an NIP status query (TSQ) followed by a refund or late credit, or a late wallet credit.

**Rules modelled:**
- CBN cash-out caps: ₦100k per customer per day, ₦1.2m per agent per day and ₦500k per customer per week
- the 70 m geo-fence at the PTSA
- KYC tier limits
- the agent picking the wrong account type at the terminal (codes 52/53)
- PIN tries exceeded (75)
- OTP failures

**Response codes** are in `codes.py`: ISO 8583 DE39 for card transactions, NIBSS NIP codes for transfers, and a LOCAL set for CBN cap rejections. Each code has one owner. v1's misuses (00 for an invalid card, 11 for a cloned card, and so on) are gone, and a test guards against them coming back.

## Outputs

```
runs/<name>/observable/spans.parquet         one row per hop per message (OpenTelemetry-style)
runs/<name>/observable/transactions.parquet  one row per attempt, as the customer sees it
runs/<name>/hidden/ground_truth.parquet      injected root cause per failed attempt
runs/<name>/hidden/episodes.parquet          every fault episode
runs/<name>/provenance.json                  where every parameter came from; [E] estimates listed
```

Models may only read `observable/`. A test checks that no ground-truth column reaches the features.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m remsimer2 simulate --days 2 --volume 15000 --out runs/demo
.venv/bin/python -m remsimer2 validate --days 2 --volume 15000             # stylised facts vs NIBSS/CBN
.venv/bin/python -m remsimer2 localise --days 3 --volume 20000 --scenario stress
.venv/bin/python -m pytest tests
```

All parameters live in `params/default.yaml`. Each one is a `{value, prov}` pair: a citation, or a tag of [E] estimate, [S] snippet-level source or [C] company claim.

## Current results (seeds 1/2, 11 Nov 2026)

**Validation** (`validate --days 2 --volume 15000`, seeds 1 and 2): all six stylised facts pass.

| Fact | Simulated | Target |
|---|---|---|
| NIP:POS volume ratio | 7.8–8.0 | ~7.7, NIBSS 2024 [S] |
| POS card failure | 13.4–14.4% | 13–15%, NIBSS [S] |
| NIP failure | 0.85–1.1% | ~0.7%, NIBSS [S] |
| POS mean ticket | ~₦9.6k | ~₦12.5k [S] |
| Failures over-dispersed in time | yes (dispersion index ~2) | bursty outages |

The failure rates match because the [E] parameters were tuned to these bands. That makes this a calibration, not independent evidence.

**Localisation.** The model trains on seed 1 and is tested on seed 2, a different population and fault history. Only technical failures are scored.

| Scenario | Model | AC@1 | AC@3 | macro-F1 | AC@1 on DEGRADED (slow, not dead) |
|---|---|---|---|---|---|
| default (4 days) | always guess the most common role | 0.48 | — | 0.07 | — |
| default | rule: last hop that answered | 0.83 | — | 0.53 | 0.65 |
| default | gradient boosting | 0.96 | 0.96 | 0.57 | 0.96 (n=54) |
| stress (3 days) | rule: last hop that answered | 0.68 | — | 0.51 | 0.36 |
| stress | gradient boosting | 0.99 | 1.00 | 0.78 | 0.99 |

**How to read these numbers:**
- Outages (DOWN) are easy, because the missing span gives them away. The rule scores 100% on them.
- The hard and interesting case is a DEGRADED entity, where a slow issuer makes the switch time out. There the rule falls to 36% and the model reaches 99%.
- The simulator's own signatures are clean (deterministic nested timeouts, complete spans), so **these accuracies overstate real-world performance**. The next steps below make the task harder on purpose.

## Next steps (phased plan in the report)

1. **Make the task harder:**
   - drop a share of spans to mimic tracing gaps
   - test what each party can see on its own: merchant/PTSP, switch, customer
   - train on one fault mix and test on an unseen one
   - test when two faults happen at once
2. **Explain predictions with SHAP** (needs LightGBM), and add a pgmpy Bayesian network over the known route graph.
3. **Let predictions drive decisions:** retry, fail over to the other PTSA, trigger an early TSQ. Then measure the drop in debited-but-failed counts.
4. **Add realism:**
   - encode messages with `pyiso8583`
   - port the NIP failure cases from `donaina/Fineract-NIP-Simulator`
   - add an ISO 20022 / NPS status layer
5. **Calibrate with real data:** a short survey of Moniepoint, OPay and PalmPay agents in Ile-Ife, plus a SALib sensitivity sweep over every [E] parameter.
6. **A Streamlit "where did my money go" dashboard.**

## Known assumptions to state in the thesis

- No public per-entity MTBF/MTTR, per-code failure shares or Nigerian hourly volume curve exists, so these are all [E].
- How OPay and PalmPay card traffic is switched, and which processor acquires for each gateway, are assumptions.
- Whether the ₦1.2m agent cap applies to transfer-funded cash-outs (route B) is unknown, so it is not applied.
- NIP codes come from a developer gist, not an official NIBSS document. The scheme-specific cryptogram codes (Mastercard 88, Visa 82) are not used until they are verified.
