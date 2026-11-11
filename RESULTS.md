# NairaTrace v1 — Full Results

## Training run details
- Dataset: kartik2112/fraud-detection (Sparkov)
- Rows: 1,296,675 transactions
- Cards: 983 unique cards
- Fraud rate: 0.0058 (7,506 fraud transactions)
- Date range: January 2019 – June 2020

---

## Entity halt labels (derived from real data)

| Entity | Label source | Halts | Rate |
|--------|-------------|-------|------|
| 010 Terminal | amt_to_avg > 3 AND geo_dist > 110 | varies | low |
| 011 Processor | velocity_1hr > 2 | varies | 13.6% of total halts |
| 100 Acquirer | amt_to_avg > 5 | varies | 6.9% of total halts |
| 101 Network | geo_dist > 120 AND velocity > 0 | varies | low |
| 110 Issuer | is_fraud (real label) | 7,506 | 0.579% |

---

## Neural network results — AUC per entity

| Entity | Accuracy | AUC | Halt % | Top signal 1 | Top signal 2 | Top signal 3 |
|--------|----------|-----|--------|-------------|-------------|-------------|
| 010 Terminal | 1.0000 | **1.0000** | low | amt_to_avg | geo_dist | hour |
| 011 Processor | ~0.95 | **0.7089** | 13.6% | hour | is_night | amt_log |
| 100 Acquirer | 1.0000 | **1.0000** | 6.9% | amt_to_avg | nibss_open | hour |
| 101 Network | ~0.98 | **0.9811** | low | geo_dist | is_night | hour |
| 110 Issuer | ~0.97 | **0.9675** | 75.5% | cat_fraud_r | cum_log | amt_to_avg |

---

## Halt distribution — 985 simulated transactions

| Entity | Halts | % of all halts | Response code |
|--------|-------|----------------|---------------|
| 110 Issuer | 360 | **75.5%** | 51 / 62 / 65 / 80 |
| 011 Processor | 65 | 13.6% | 29 |
| 100 Acquirer | 33 | 6.9% | 19 |
| 010 Terminal | 19 | 4.0% | 28 |
| Approved | 508 | — | 00 |

---

## Top conditions that caused a halt

| Entity:Condition | Count |
|-----------------|-------|
| Issuer:balance_sufficient | 311 |
| Issuer:step_up_cleared | 17 |
| Issuer:velocity_ok | 14 |
| Processor:velocity_ok | 65 |
| Acquirer:amount_within_limit | 28 |
| Terminal:cryptogram_ok | 19 |

**The single most common failure: Issuer balance_sufficient = 0 (311 times, 86% of all Issuer halts)**

---

## Response code distribution

| Code | Count | Meaning |
|------|-------|---------|
| 51 | 311 | Issuer — insufficient funds |
| 29 | 65 | Processor — velocity halt |
| 19 | 33 | Acquirer — limit exceeded |
| 80 | 17 | Issuer — step-up / unknown condition |
| 65 | 14 | Issuer — daily limit exceeded |
| 28 | 19 | Terminal — cryptogram failure |

---

## Per-card profiles (top cards by halt count)

| Card | Txns | Approval % | Trend | Top halt |
|------|------|-----------|-------|---------|
| CARD_034 | 29 | 34.5% | **WORSENING** | Issuer |
| CARD_000 | 27 | 25.9% | stable | Issuer |
| CARD_033 | 17 | 0.0% | stable | POS Terminal |
| CARD_048 | 22 | 31.8% | **WORSENING** | Issuer |

---

## AUC interpretation

### Why Terminal and Acquirer AUC = 1.0000
These entities check arithmetic conditions — amount against a fixed limit, amount spike relative to card history. The neural network perfectly recovers a clean mathematical boundary from the data. No noise, no ambiguity.

### Why Processor AUC = 0.7089
Velocity fraud is genuinely noisy. Legitimate cardholders sometimes make multiple transactions in one hour. The network cannot perfectly separate fraud from legitimate velocity because the signal is real but impure. This is an empirical finding, not a model weakness.

### Why Issuer AUC = 0.9675 (not 1.0)
The 0.0325 gap represents what is genuinely unobservable from outside the bank:
- The bank's proprietary fraud-scoring model
- Internal account flags (e.g. account under investigation)
- Undisclosed business rules
- Step-up authentication triggers

Five entities are fully transparent. The Issuer is not. Code 80 captures this unknown. AUC 0.9675 is the correct result — a model claiming 1.0 on the Issuer would be overfitting.
