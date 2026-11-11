"""Load a params YAML file whose leaves are {value, prov} pairs.

`load()` returns (values, provenance): two nested dicts with the same keys,
one holding the values and one holding where each value came from.
"""

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "params" / "default.yaml"


def _split(node):
    if isinstance(node, dict) and "value" in node and "prov" in node:
        return node["value"], node["prov"]
    if isinstance(node, dict):
        values, provs = {}, {}
        for k, v in node.items():
            values[k], provs[k] = _split(v)
        return values, provs
    raise ValueError(f"Parameter leaf without value/prov: {node!r}")


def load(path=DEFAULT_PATH, overrides=None):
    with open(path) as f:
        raw = yaml.safe_load(f)
    values, provs = _split(raw)
    for dotted, val in (overrides or {}).items():
        node = values
        *parents, last = dotted.split(".")
        for key in parents:
            node = node[key]
        node[last] = val
    return values, provs


def estimates(provs, prefix=""):
    """List every parameter path whose provenance is tagged [E]."""
    out = []
    for k, v in provs.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out += estimates(v, path + ".")
        elif "[E]" in str(v):
            out.append(path)
    return out
