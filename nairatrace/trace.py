"""Recorder for observable logs and hidden ground truth.

Observable (what switches, banks and terminals would log):
  spans         one row per hop per message, OpenTelemetry-style
  transactions  one row per attempt, as a customer/merchant would see it
Hidden (only the simulator knows):
  ground_truth  the injected root cause of every failed attempt
  episodes      every fault episode (written by entities)

The localisers in localise.py may read only the observable tables.
"""

import itertools

SPAN_STATUSES = ("OK", "DECLINE", "TIMEOUT", "ERROR", "NO_RESPONSE")


class Tracer:
    def __init__(self):
        self.spans = []
        self.txs = []   # Transaction objects; rows are built at export so late reversals are included
        self._span_ids = itertools.count(1)
        self._trace_ids = itertools.count(1)

    def new_trace_id(self):
        return f"TR{next(self._trace_ids):08d}"

    def new_span_id(self):
        return next(self._span_ids)

    def span(self, tx, span_id, parent_id, entity_fqn, role, op, mti, t_start, t_end,
             status, code, space):
        assert status in SPAN_STATUSES, status
        self.spans.append({
            "trace_id": tx.trace_id, "span_id": span_id, "parent_span_id": parent_id,
            "entity": entity_fqn, "entity_type": role, "op": op, "mti": mti,
            "t_start": t_start, "t_end": t_end, "latency_ms": round((t_end - t_start) * 1000, 1),
            "status": status, "code": code, "code_space": space,
            "retry_no": tx.retry_no, "stan": tx.stan, "rrn": tx.rrn,
        })

    def finish(self, tx):
        self.txs.append(tx)

    def transaction_rows(self):
        return [tx.observable_row() for tx in self.txs]

    def ground_truth_rows(self):
        return [tx.truth_row() for tx in self.txs]
