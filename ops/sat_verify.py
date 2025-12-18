# ops/sat_verify.py
from __future__ import annotations

from typing import Any, Dict

from . import register_op


def _lit_value(lit: int, bits: str) -> bool:
    """Evaluate one literal under an assignment_bits string."""
    v = abs(int(lit))
    if v <= 0:
        return False
    if v > len(bits):
        # If assignment doesn't cover this variable, treat as False
        return False
    var_true = (bits[v - 1] == "1")
    return var_true if lit > 0 else (not var_true)


@register_op("sat_verify")
def sat_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify CNF satisfaction for a proposed assignment.

    payload:
      - cnf: List[List[int]]
      - assignment_bits: str of '0'/'1'
    """
    cnf = payload.get("cnf")
    bits = payload.get("assignment_bits")

    if not isinstance(bits, str) or any(c not in "01" for c in bits):
        raise ValueError("assignment_bits must be a string of 0/1")

    if not isinstance(cnf, list) or any(not isinstance(cl, list) for cl in cnf):
        raise ValueError("cnf must be a list of clauses (list[list[int]])")

    # Determine max var index used in cnf
    max_var = 0
    for clause in cnf:
        for lit in clause:
            try:
                max_var = max(max_var, abs(int(lit)))
            except Exception:
                raise ValueError("cnf literals must be ints")

    # Verify each clause
    for idx, clause in enumerate(cnf):
        clause_sat = False
        for lit in clause:
            if _lit_value(int(lit), bits):
                clause_sat = True
                break
        if not clause_sat:
            return {
                "sat": False,
                "unsat_clause": idx,
                "nvars": max_var,
                "nclauses": len(cnf),
            }

    return {
        "sat": True,
        "unsat_clause": None,
        "nvars": max_var,
        "nclauses": len(cnf),
    }
