"""preprocessor.py — backward-compatible re-export of core.features.preprocess."""

from core.features import STOPWORDS_PRESERVED_CONTRACTION_MAP, preprocess

__all__ = ["STOPWORDS_PRESERVED_CONTRACTION_MAP", "preprocess"]
