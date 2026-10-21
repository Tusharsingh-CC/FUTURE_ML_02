"""Text cleaning for support tickets.

The same function is used at training time and at prediction time, so the model
always sees text that was cleaned in exactly the same way.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations carry meaning in support tickets ("not working" != "working"),
# so we deliberately keep them even though they are normally "stop words".
KEEP_WORDS = {"not", "no", "nor", "never", "cannot", "without", "against"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PLACEHOLDER = re.compile(r"\{[^}]*\}")  # template leftovers like {product_purchased}
_NON_ALPHA = re.compile(r"[^a-z\s]")  # also drops numbers
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip noise, normalise contractions, remove stop words."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _PLACEHOLDER.sub(" ", t)

    # normalise contractions so "can't" and "cannot" become the same token
    t = t.replace("won't", "will not").replace("can't", "cannot")
    t = re.sub(r"n't\b", " not", t)

    t = _NON_ALPHA.sub(" ", t)
    tokens = [w for w in _SPACES.split(t) if len(w) > 1 and w not in STOP_WORDS]
    return " ".join(tokens)
