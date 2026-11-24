"""
preprocessor.py
---------------
Centralised text-preprocessing module for EmotiSense AI.
Matches Emotion_Heart_Validation.ipynb exactly:
  1. Lowercase
  2. Strip URLs and HTML tags (for safe live web inputs)
  3. Strip punctuation (string.punctuation)
  4. Keep ASCII non-digit characters only
  5. Tokenize word boundaries and rejoin with single spaces
  6. Stopwords and negation tokens (e.g. 'not', 'never', 'no', 'didnt') are
     intentionally preserved as validated in Emotion_Heart_Validation.ipynb.
"""

import re
import string

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"\b\w+\b")


def preprocess(text: str) -> str:
    """Clean and normalize raw text while retaining negation and stopwords."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(_PUNCT_TRANS)
    text = "".join(c for c in text if c.isascii() and not c.isdigit())
    return " ".join(_WORD_RE.findall(text))


def batch_preprocess(texts):
    """Apply preprocess() to an iterable of raw strings."""
    return [preprocess(t) for t in texts]
