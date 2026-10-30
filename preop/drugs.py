"""Reading the record (brand -> generic). This only *identifies* drugs; it adds no policy criteria."""

from __future__ import annotations

import re

# canonical generic -> names it may appear under
_ANTICOAGULANTS: dict[str, tuple[str, ...]] = {
    "warfarin": ("warfarin", "coumadin", "jantoven"),
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "dabigatran": ("dabigatran", "pradaxa"),
    "edoxaban": ("edoxaban", "savaysa", "lixiana"),
    "enoxaparin": ("enoxaparin", "lovenox"),
    "dalteparin": ("dalteparin", "fragmin"),
    "fondaparinux": ("fondaparinux", "arixtra"),
    "heparin": ("heparin",),
}
# Antiplatelets are NOT anticoagulants for Rule 3 (confirmed by the sample labels).
ANTIPLATELETS = ("aspirin", "asa", "clopidogrel", "plavix", "prasugrel", "ticagrelor", "dipyridamole")

_LOOKUP = {name: canon for canon, names in _ANTICOAGULANTS.items() for name in names}
_PATTERN = re.compile(r"\b(" + "|".join(sorted(_LOOKUP, key=len, reverse=True)) + r")\b", re.I)


def canonical_anticoagulant(text: str | None) -> str | None:
    """'Coumadin 5 mg PO daily' -> 'warfarin'; 'ASA 81' -> None."""
    if not text:
        return None
    m = _PATTERN.search(text)
    return _LOOKUP[m.group(1).lower()] if m else None


def find_anticoagulants(text: str) -> list[str]:
    return sorted({_LOOKUP[m.group(1).lower()] for m in _PATTERN.finditer(text)})
