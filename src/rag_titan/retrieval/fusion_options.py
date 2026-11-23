"""
The two options of RRF fusion (see fusion.py), kept apart so each file stays short:

    check_weight():   one weight must be a number above 0.
    check_weights():  read {list name: weight}; a list left out counts 1.0.
    keep_list_tops(): make sure every list's #1 chunk is in the fused top_k.
"""

import math


def check_weight(weight, name):
    """
    Return the weight as a float, or raise a clear ValueError.

    Args:
        weight: a number above 0 (1.0 = a normal vote, 2.0 = twice as much).
        name: what the weight is called in the error, e.g. "dense_weight".
    """
    is_number = isinstance(weight, (int, float)) and not isinstance(weight, bool)
    if not is_number or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{name} must be a number above 0, got {weight!r}")
    return float(weight)


def check_weights(weights, names):
    """
    Return one weight per list name; a name left out counts 1.0.

    Args:
        weights: None (every list counts 1.0) or {name: number above 0}.
        names: the names of the lists being fused.
    """
    if weights is None:
        return {name: 1.0 for name in names}
    if not isinstance(weights, dict):
        raise ValueError(f"weights must be a dict like {{'dense': 2.0}}, got {weights!r}")

    unknown = [name for name in weights if name not in names]
    if unknown:
        raise ValueError(f"weights name unknown lists {unknown}; the lists are {list(names)}")
    for name, weight in weights.items():
        check_weight(weight, f"the weight of {name!r}")
    return {name: float(weights.get(name, 1.0)) for name in names}


def keep_list_tops(matches, lists, top_k):
    """
    The top_k of `matches`, but with every list's #1 chunk in it.

    A #1 that fell below the top_k takes the place of the weakest chunk that is
    not a #1 itself; its score_trace gets "kept_as_top_of": [list names]. With
    more lists than top_k, the #1s with the best fused scores win. A #1 dropped
    as a near-copy is not brought back: its passage is already there.

    Returns:
        top_k chunk dicts, still best fused score first.
    """
    tops = {}
    for name, results in lists.items():
        if results:
            tops.setdefault(results[0]["id"], []).append(name)

    # matches are best first, so the strongest #1s come first
    protected = [match["id"] for match in matches if match["id"] in tops][:top_k]
    others = [match["id"] for match in matches if match["id"] not in protected]
    chosen = set(protected + others[:top_k - len(protected)])
    usual = {match["id"] for match in matches[:top_k]}

    kept = []
    for match in matches:
        if match["id"] not in chosen:
            continue
        if match["id"] not in usual:
            match = {**match, "score_trace": {**match["score_trace"],
                                              "kept_as_top_of": tops[match["id"]]}}
        kept.append(match)
    return kept
