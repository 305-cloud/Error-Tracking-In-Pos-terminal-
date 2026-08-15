# REMSIMER — Six-Entity Neural Network State Machine
### Development of an Error Tracking System for POS Systems

> **"When a POS transaction fails, the customer is left in the dark. REMSIMER pinpoints the exact entity and exact condition that caused the halt."**

---

## Overview

REMSIMER is a machine learning model that tracks the flow of a card transaction through six entities in a payment system and isolates the exact point of failure. It was developed as a final year project at Obafemi Awolowo University, Ile-Ife.

The model answers one question precisely:

**At what point in the system is there a fault, and what exactly caused it?**

---

## The Problem

A card transaction passes through six separate entities before it is approved or declined:

```
Customer → Terminal → Processor → Acquirer → Card Network → Issuer
```

When a transaction fails, existing systems return a generic decline code. The customer does not know whether the problem was:
- Their card number being invalid
- The terminal's cryptogram failing
- The processor rejecting it for velocity
- The acquirer's limit being exceeded
- The card network detecting a clone
- The issuer declining for insufficient funds

REMSIMER tracks the transaction through every entity and records the exact condition that caused the halt.

---

## Architecture

```
Prompt → INPUT → [64 → 32 → 16 Hidden Layers] → True(pass) / False(halt)
         ↓
       InputX₀ → 000 Customer    → InputX₁
                 010 Terminal    → InputX₂
                 011 Processor   → InputX₃
                 100 Acquirer    → InputX₄
                 101 Network     → InputX₅
                 110 Issuer      → APPROVED code 00
                                   OR HALT code 51/62/65/80
```

Each entity runs an **AND gate**. Every condition must equal 1 to pass. The moment any condition is 0, the chain halts and the fault is recorded.

The output of entity N (called **InputX**) becomes the input of entity N+1 — carrying the full audit trail forward.

---

## Response Codes

| Entity | ID | Halt Code | Meaning |
|--------|-----|-----------|---------|
| Customer | 000 | 00 | Invalid card number |
| POS Terminal | 010 | 28 | Cryptogram / expiry failure |
| Processor | 011 | 29 | Velocity / routing failure |
| Acquirer | 100 | 19 | Limit exceeded |
| Card Network | 101 | 11 | Possible card clone |
| Issuer | 110 | 51 | Insufficient funds |
| Issuer | 110 | 62 | Card blocked |
| Issuer | 110 | 65 | Daily limit exceeded |
| Issuer | 110 | 80 | Step-up / unknown private condition |

---

## Key Mathematics

**Customer — Luhn checksum (PAN: 3525668150756629)**
```
Step 1: Remove check digit (9). Reverse: [2,6,6,5,7,0,5,1,8,6,6,5,2,5,3]
Step 2: Double every other digit: [4,6,3,5,5,0,1,1,7,6,3,5,4,5,6]
Step 3: Sum + check digit: 61 + 9 = 70
Step 4: 70 mod 10 = 0 → VALID
```

**Terminal — ARQC cryptogram**
```
ARQC = AES₁₂₈(PAN ‖ amount ‖ ATC ‖ UN ‖ K_chip)
A cloned card cannot produce a matching ARQC — it does not have K_chip.
```

**Processor — Velocity**
```
v = |{t ∈ window : Δt ≤ 3600s}|    v > 2 → halt (code 29)
```

**Network — Haversine impossible travel**
```
a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
d = 2·6371·arcsin(√a) km
d > 120 km AND velocity > 0 → clone detected (code 11)
```

---

## Results

Trained on **1,296,675 real card transactions** (Sparkov dataset, Kaggle).

### AUC scores per entity

| Entity | AUC | Key signal learned |
|--------|-----|--------------------|
| 010 POS Terminal | **1.0000** | amt_to_avg + geo_dist |
| 011 Processor | **0.7089** | velocity + cat_fraud_r |
| 100 Acquirer | **1.0000** | amt_to_avg + nibss_open |
| 101 Card Network | **0.9811** | geo_dist + is_night |
| 110 Issuer | **0.9675** | cat_fraud_r + cum_log |

### Halt distribution (985 transactions)

| Entity | Halts | % of all halts |
|--------|-------|----------------|
| 110 Issuer | 360 | **75.5%** |
| 011 Processor | 65 | 13.6% |
| 100 Acquirer | 33 | 6.9% |
| 010 Terminal | 19 | 4.0% |

**75.5% of all halts happen at the Issuer. Within those, 86% are caused by one condition: `balance_sufficient = 0`.**

### The central finding — Issuer AUC gap

The Issuer AUC of **0.9675** is not a weakness. The 0.0325 gap represents what is genuinely unobservable from outside the bank: proprietary fraud-scoring models, undisclosed business rules, and internal account flags. Five entities are fully transparent. The Issuer is not. Code 80 captures this unknown.

### Fault isolation example

```
Entity           Gate    Conditions
──────────────────────────────────────────────────────────────
Customer         OPEN    card_number_valid=OK  account_exists=OK
POS Terminal     OPEN    cryptogram_ok=OK  expiry_valid=OK  pin_encrypted=OK
Processor        OPEN    nibss_routable=OK  velocity_ok=OK  merchant_ok=OK
Acquirer         OPEN    amount_within_limit=OK  agent_cbn_ok=OK  nibss_open=OK
Card Network     OPEN    bin_valid=OK  cryptogram_verified=OK  member_ok=OK
Issuer           HALT    balance_sufficient=FAIL  ← fault isolated here

HALTED at Issuer — code 51 — 'balance_sufficient' = 0
```

---

## Nigerian Payment Context

The model is calibrated for the Nigerian payment ecosystem:

- **NIBSS** routing validation in the Processor entity
- **CBN agent banking** cap of ₦50,000 enforced in the Acquirer entity
- **NUBAN** account number validation for NIP transfers
- **Category fraud rates** from NIBSS e-Payment Factsheet 2023 and CBN Annual Report 2023
- **Verve / Mastercard / Visa** daily limits enforced at the Issuer

---

## Two Models

### 1. POS Model (`remsimer_pos.py`)
Six-entity POS pipeline. Card physically present. ARQC cryptogram. PIN encryption.

### 2. Online Payment Model (`remsimer_online.py`)
Three payment types — CNP (card-not-present), NIP (bank transfer), WALT (fintech wallet).
Terminal replaced by Auth Layer (OTP + 3DS + device score).
Payment Gateway added as a new entity (Paystack / Flutterwave).

---

## Files

```
REMSIMER/
│
├── remsimer_pos.py              # POS model — full code
├── remsimer_online.py           # Online payment model — CNP / NIP / WALT
│
├── REMSIMER_POS_colab.ipynb     # Colab notebook — POS model (copy-paste cells)
├── REMSIMER_online_colab.ipynb  # Colab notebook — online model
│
├── REMSIMER_POS_report.pdf      # 6-page technical report with results
│
└── README.md
```

---

## How to Run

### Option A — Local

```bash
pip install pandas numpy scikit-learn
python remsimer_pos.py
```

Update `DATA_PATH` in the file to point to your `fraudTrain.csv`.

### Option B — Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `REMSIMER_POS_colab.ipynb`
3. Run cells 1 to 19 in order
4. Upload `fraudTrain.csv` when Cell 2 prompts

### Training data

Download `fraudTrain.csv` from Kaggle:
[kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

1,296,675 real card transactions, January 2019 – June 2020.

---

## Per-Card Trend Detection

REMSIMER tracks every card across its full transaction history and detects whether performance is worsening, improving, or stable.

```
Card         Txns  Appr%       Trend  Top halt entity
────────────────────────────────────────────────────
CARD_034       29  34.5%   worsening  Issuer
CARD_000       27  25.9%      stable  Issuer
CARD_033       17   0.0%      stable  POS Terminal
```

CARD_033 — 100% halt rate at Terminal, `cryptogram_ok = 0`. Chip is broken.
CARD_034 — Halt rate rising. Worsening trend detected.

---

## Academic Details

| | |
|---|---|
| **Title** | Development of an Error Tracking System for POS System |
| **Author** | Mosobalaje Muiz Ayomide (CSC/2017/086) |
| **Supervisor** | Dr. A.R. Iyanda |
| **Institution** | Obafemi Awolowo University, Ile-Ife |
| **Year** | 2024 |

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — MLPClassifier, StandardScaler, train_test_split
- **pandas / numpy** — data loading, feature engineering
- **math** — Haversine distance, Luhn checksum
- **Training data** — kartik2112/fraud-detection (Kaggle / Sparkov)

---

## License

MIT License — free to use, modify, and distribute with attribution.
