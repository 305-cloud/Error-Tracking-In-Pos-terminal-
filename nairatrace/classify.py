"""Readable classifications added to every transaction.

Both are built from observable fields only (the code the agent sees and the
time), so the same functions can label real agent data collected in the field.

Outcome category, from the response code:
  Successful            00, or a transfer that completed after a status query
  Customer issue        wrong PIN, no money, expired/invalid card, failed OTP, wrong account number
  Agent or terminal     wrong account type selected, terminal outside its geo-fence, invalid merchant
  Limit reached         card/transfer limits and CBN cash-out caps
  Bank decline          do not honour, not permitted, suspected fraud, dormant account
  Technical failure     switch/bank/NIBSS errors, timeouts, no response at all
  Debited but failed    the customer lost money for a failed transaction (overrides the above)

Time of transaction:
  datetime, part_of_day, is_month_end (25th-1st, salary days)
"""

import pandas as pd

CATEGORY = {
    "CARD": {
        "00": "Successful",
        "14": "Customer issue", "51": "Customer issue", "54": "Customer issue",
        "55": "Customer issue", "75": "Customer issue",
        "03": "Agent or terminal", "52": "Agent or terminal", "53": "Agent or terminal", "58": "Agent or terminal",
        "61": "Limit reached", "65": "Limit reached",
        "05": "Bank decline", "57": "Bank decline", "59": "Bank decline",
        "15": "Technical failure", "30": "Technical failure", "68": "Technical failure", "91": "Technical failure",
        "92": "Technical failure", "94": "Technical failure", "96": "Technical failure",
    },
    "NIP": {
        "00": "Successful",
        "07": "Customer issue", "51": "Customer issue",
        "61": "Limit reached",
        "06": "Bank decline", "34": "Bank decline", "57": "Bank decline",
        "09": "Technical failure", "16": "Technical failure", "26": "Technical failure", "68": "Technical failure",
        "91": "Technical failure", "96": "Technical failure", "97": "Technical failure",
    },
    "LOCAL": {
        "00": "Successful",
        "L01": "Limit reached", "L02": "Limit reached", "L03": "Limit reached",
        "L10": "Customer issue",
    },
}
CATEGORIES = ["Successful", "Customer issue", "Agent or terminal", "Limit reached",
              "Bank decline", "Technical failure", "Debited but failed"]

PARTS_OF_DAY = [(0, 6, "Night"), (6, 12, "Morning"), (12, 16, "Afternoon"), (16, 20, "Evening"), (20, 24, "Late evening")]


def outcome_category(row):
    if row["customer_debited"]:
        return "Debited but failed"
    if row["final_status"] in ("APPROVED", "APPROVED_LATE"):
        return "Successful"
    code, space = row["final_code"], row["code_space"]
    if pd.isna(code):
        return "Technical failure"   # no response code at all
    return CATEGORY.get(space, {}).get(code, "Technical failure")


def part_of_day(hour):
    for lo, hi, name in PARTS_OF_DAY:
        if lo <= hour < hi:
            return name
    raise ValueError(hour)


def add_columns(transactions, start_date):
    t = transactions.copy()
    dt = pd.Timestamp(start_date) + pd.to_timedelta(t.ts, unit="s")
    t["datetime"] = dt.dt.floor("s")
    t["part_of_day"] = dt.dt.hour.map(part_of_day)
    t["is_month_end"] = (dt.dt.day >= 25) | (dt.dt.day == 1)
    t["outcome_category"] = t.apply(outcome_category, axis=1)
    return t
