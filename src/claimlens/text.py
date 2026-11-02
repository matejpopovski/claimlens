"""Tokenisation for the lexical index.

Scientific text defeats a plain word regex in two specific ways: Greek letters
appear as both symbols and names ("TNF-α" and "TNF-alpha" are the same entity),
and short alphanumeric gene names ("p53", "IL-6") must survive intact. Everything
here exists to keep those matches.
"""

from __future__ import annotations

import re
import unicodedata

_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "µ": "mu",
    "ν": "nu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}
_GREEK_TABLE = str.maketrans({k: f" {v} " for k, v in _GREEK.items()})
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lucene's default English stop set, so the lexical baseline is comparable with
# the Elasticsearch English analyzer that BEIR's published BM25 numbers use.
STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the
    their then there these they this to was will with""".split()
)


def s_stem(token: str) -> str:
    """Harman's S-stemmer: strip plurals only, which is safe for biomedical terms.

    A full Porter stemmer conflates more aggressively than gene and drug names tolerate.
    """
    if len(token) > 3 and token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def tokenize(text: str, *, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).translate(_GREEK_TABLE).lower()
    tokens = _TOKEN_RE.findall(normalized)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [s_stem(t) for t in tokens]
    return tokens
