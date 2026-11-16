"""Response-code tables for NairaTrace v2.

Two separate code spaces (see report, "Corrected response codes"):

* CARD - ISO 8583:1987 field 39 (DE39), as used by Interswitch/Postilion-style
  switches. Codes confirmed on Interswitch docs are tagged [ISW]; the rest are
  standard ISO meanings [ISO] and still need checking against the NIBSS POS
  DE39 spec.
* NIP  - NIBSS Instant Payment status codes. Same numbers, different meanings
  (e.g. NIP 07 = invalid account). Source: developer gist, not an official
  NIBSS document [GIST].

LOCAL holds rejections that never leave the principal's app or the PTSA
(CBN cash-out caps, geo-fence), so they get their own namespace instead of
being forced into DE39.

Each code has exactly one owner role: the entity that makes the decision.
`kind` is "business" (a decision about the customer/account) or "technical"
(an infrastructure problem). `outcome` is "approved", "declined" or
"unknown" - unknown means the customer may already be debited and a
reversal (card, MTI 0420) or TSQ (NIP) must follow.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Code:
    code: str
    meaning: str
    owner: str      # role that decides: ISSUER, SWITCH, ACQUIRER, PTSA, ...
    kind: str       # business | technical | none
    outcome: str    # approved | declined | unknown
    source: str


def _table(rows):
    return {r[0]: Code(*r) for r in rows}


CARD = _table([
    ("00", "Approved", "ISSUER", "none", "approved", "[ISW]"),
    ("03", "Invalid merchant", "ACQUIRER", "business", "declined", "[ISO]"),
    ("05", "Do not honour", "ISSUER", "business", "declined", "[ISO]"),
    ("14", "Invalid card number", "ISSUER", "business", "declined", "[ISO]"),
    ("15", "No such issuer", "SWITCH", "technical", "declined", "[ISO]"),
    ("30", "Format error", "ACQUIRER", "technical", "declined", "[ISO]"),
    ("51", "Insufficient funds", "ISSUER", "business", "declined", "[ISW]"),
    ("52", "No current account", "ISSUER", "business", "declined", "[ISO]"),
    ("53", "No savings account", "ISSUER", "business", "declined", "[ISO]"),
    ("54", "Expired card", "ISSUER", "business", "declined", "[ISO]"),
    ("55", "Incorrect PIN", "ISSUER", "business", "declined", "[ISW]"),
    ("57", "Transaction not permitted to cardholder", "ISSUER", "business", "declined", "[ISW]"),
    ("58", "Transaction not permitted on terminal", "PTSA", "business", "declined", "[ISO]"),
    ("59", "Suspected fraud", "ISSUER", "business", "declined", "[ISO]"),
    ("61", "Exceeds withdrawal amount limit", "ISSUER", "business", "declined", "[ISO]"),
    ("65", "Exceeds withdrawal frequency limit", "ISSUER", "business", "declined", "[ISW]"),
    ("68", "Response received too late", "SWITCH", "technical", "unknown", "[ISO]"),
    ("75", "PIN tries exceeded", "ISSUER", "business", "declined", "[ISO]"),
    ("91", "Issuer or switch inoperative", "SWITCH", "technical", "declined", "[ISW]"),
    ("92", "Routing error", "SWITCH", "technical", "declined", "[ISO]"),
    ("94", "Duplicate transmission", "SWITCH", "technical", "declined", "[ISO]"),
    ("96", "System malfunction", "SWITCH", "technical", "unknown", "[ISW]"),
])

NIP = _table([
    ("00", "Approved or completed successfully", "NIBSS", "none", "approved", "[GIST]"),
    ("06", "Dormant account", "BENEFICIARY_FI", "business", "declined", "[GIST]"),
    ("07", "Invalid account", "BENEFICIARY_FI", "business", "declined", "[GIST]"),
    ("08", "Account name mismatch", "BENEFICIARY_FI", "business", "declined", "[GIST]"),
    ("09", "Request processing in progress", "NIBSS", "technical", "unknown", "[GIST]"),
    ("16", "Unknown bank code", "NIBSS", "technical", "declined", "[GIST]"),
    ("26", "Duplicate record", "NIBSS", "technical", "declined", "[GIST]"),
    ("34", "Suspected fraud", "BENEFICIARY_FI", "business", "declined", "[GIST]"),
    ("51", "No sufficient funds", "SENDER_FI", "business", "declined", "[GIST]"),
    ("57", "Transaction not permitted to sender", "SENDER_FI", "business", "declined", "[GIST]"),
    ("61", "Transfer limit exceeded", "SENDER_FI", "business", "declined", "[GIST]"),
    ("68", "Response received too late", "NIBSS", "technical", "unknown", "[GIST]"),
    ("91", "Beneficiary bank not available", "NIBSS", "technical", "unknown", "[GIST]"),
    ("96", "System malfunction", "NIBSS", "technical", "unknown", "[GIST]"),
    ("97", "Timeout waiting for response from destination", "NIBSS", "technical", "unknown", "[GIST]"),
])

LOCAL = _table([
    ("00", "Agent wallet credited", "PRINCIPAL", "none", "approved", "principal ledger"),
    ("L01", "CBN cap: N100k per customer per day at agent", "PRINCIPAL", "business", "declined", "CBN 17 Dec 2024"),
    ("L02", "CBN cap: N1.2m per agent per day", "PRINCIPAL", "business", "declined", "CBN 17 Dec 2024"),
    ("L03", "CBN cap: N500k per customer per week", "ISSUER", "business", "declined", "CBN 17 Dec 2024"),
    ("L10", "3DS/OTP: Token Authentication failed", "ISSUER", "business", "declined", "Paystack support"),
])

SPACES = {"CARD": CARD, "NIP": NIP, "LOCAL": LOCAL}


def lookup(space, code):
    return SPACES[space][code]


def is_unknown_outcome(space, code):
    """Card 68/96 and NIP 09/68/96/97 leave the outcome unknown -> reversal/TSQ."""
    c = SPACES[space].get(code)
    return c is not None and c.outcome == "unknown"
