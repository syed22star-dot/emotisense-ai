"""
core/features.py
----------------
Custom domain NLP normalization, negation-aware Affective Syntax Feature Extractor,
and multi-granularity FeatureUnion builder (Word n-grams + Subword Char n-grams + Scaled Syntax)
for EmotiSense AI (scikit-learn 1.8.0).
"""

import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

STOPWORDS_PRESERVED_CONTRACTION_MAP = {
    "im": "i am",
    "ive": "i have",
    "id": "i would",
    "ill": "i will",
    "dont": "do not",
    "didnt": "did not",
    "doesnt": "does not",
    "cant": "can not",
    "cannot": "can not",
    "wont": "will not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "youre": "you are",
    "theyre": "they are",
    "weve": "we have",
    "thats": "that is",
    "theres": "there is",
    "whats": "what is",
    "hed": "he would",
    "shed": "she would",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere", "hardly", "barely",
    "scarcely", "without", "didnt", "dont", "cant", "wont", "isnt", "wasnt", "werent",
    "doesnt", "couldnt", "wouldnt", "shouldnt",
}

INTENSIFIERS = {
    "very", "so", "really", "extremely", "totally", "completely", "utterly", "super",
    "absolutely", "deeply", "truly", "incredibly", "sooo", "terribly", "awfully",
}

PIVOT_WORDS = {"but", "however", "actually", "though", "although", "yet", "instead", "still"}

FIRST_PERSON = {"i", "me", "my", "myself", "mine"}

EMOTION_LEXICON = {
    "anger": {
        "angry", "mad", "furious", "rage", "enraged", "annoyed", "irritated", "frustrated",
        "outraged", "hate", "hated", "bitter", "resentful", "offended", "hostile", "agitated",
        "infuriated", "livid", "grumpy", "cross",
    },
    "fear": {
        "afraid", "scared", "terrified", "fearful", "anxious", "nervous", "worried", "panicked",
        "panic", "frightened", "shaken", "uneasy", "tense", "dread", "paranoid", "insecure",
        "helpless", "threatened", "intimidated", "apprehensive",
    },
    "joy": {
        "happy", "joy", "joyful", "glad", "delighted", "cheerful", "relieved", "pleased",
        "thrilled", "excited", "grateful", "thankful", "blessed", "wonderful", "satisfied",
        "proud", "optimistic", "elated", "peaceful", "blissful", "great", "amazing",
    },
    "love": {
        "love", "loved", "loving", "adore", "adored", "affection", "affectionate", "caring",
        "fond", "cherish", "tender", "romantic", "devoted", "sweet", "warmth", "passionate",
        "compassionate", "sympathetic", "loyalty", "kindness",
    },
    "sadness": {
        "sad", "unhappy", "depressed", "lonely", "heartbroken", "grief", "miserable",
        "disappointed", "down", "blue", "gloomy", "hopeless", "empty", "humiliated",
        "ashamed", "guilty", "regretful", "sorrow", "crying", "hurt", "devastated",
    },
    "surprise": {
        "surprised", "shocked", "amazed", "astonished", "stunned", "startled", "speechless",
        "unexpected", "unbelievable", "baffled", "bewildered", "awe", "overwhelmed",
        "dazed", "sudden", "curious",
    },
}

POSITIVE_LEX = EMOTION_LEXICON["joy"] | EMOTION_LEXICON["love"]
NEGATIVE_LEX = EMOTION_LEXICON["anger"] | EMOTION_LEXICON["fear"] | EMOTION_LEXICON["sadness"]

SYNTAX_FEATURE_NAMES = [
    "negation_count",
    "negated_positive_cue",
    "negated_negative_cue",
    "pivot_conjunction_cue",
    "intensifier_count",
    "first_person_ratio",
    "word_count_log",
    "avg_word_len_norm",
    "lexical_diversity",
    "anger_lex_density",
    "fear_lex_density",
    "joy_lex_density",
    "love_lex_density",
    "sadness_lex_density",
    "surprise_lex_density",
]


def _collapse_elongations(token: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", token)


def preprocess(text: str) -> str:
    """
    Domain-aligned text normalization matching Emotion_Heart_Validation.ipynb:
      - lowercase + punctuation strip
      - expand common contractions so negation ('didn't' -> 'did not') is explicit
      - preserve all stopwords, pronouns, and negations
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = [STOPWORDS_PRESERVED_CONTRACTION_MAP.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def analyze_syntax_signals(raw_text: str) -> dict:
    """Compute exact 15 non-negative affective/syntax cues + matched evidence lists."""
    norm_text = preprocess(raw_text or "")
    raw_tokens = norm_text.split()
    tokens = [_collapse_elongations(t) for t in raw_tokens]
    n_tok = max(1, len(tokens))

    neg_idxs = [i for i, tok in enumerate(tokens) if tok in NEGATION_WORDS]
    intens_found = [tok for tok in tokens if tok in INTENSIFIERS]
    pivots_found = [tok for tok in tokens if tok in PIVOT_WORDS]
    fp_cnt = sum(1 for tok in tokens if tok in FIRST_PERSON)

    neg_pos_flips = []
    neg_neg_flips = []
    negated_window = set()
    for idx in neg_idxs:
        for j in range(idx + 1, min(len(tokens), idx + 4)):
            negated_window.add(j)
            w = tokens[j]
            if w in POSITIVE_LEX:
                neg_pos_flips.append(f"{tokens[idx]} ... {w}")
            if w in NEGATIVE_LEX:
                neg_neg_flips.append(f"{tokens[idx]} ... {w}")

    avg_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    lex_div = len(set(tokens)) / n_tok if tokens else 0.0

    emo_densities = {}
    for emo, lex in EMOTION_LEXICON.items():
        # Discount tokens directly negated in preceding 3-token window
        hits = sum(1 for i, tok in enumerate(tokens) if tok in lex and i not in negated_window)
        emo_densities[emo] = hits / n_tok

    vector = [
        float(len(neg_idxs)),
        float(len(neg_pos_flips)),
        float(len(neg_neg_flips)),
        float(len(pivots_found)),
        float(len(intens_found)),
        float(fp_cnt / n_tok),
        float(np.log1p(len(tokens))),
        float(avg_len / 10.0),
        float(lex_div),
        float(emo_densities["anger"]),
        float(emo_densities["fear"]),
        float(emo_densities["joy"]),
        float(emo_densities["love"]),
        float(emo_densities["sadness"]),
        float(emo_densities["surprise"]),
    ]

    return {
        "normalized_text": norm_text,
        "word_count": len(tokens),
        "vector": vector,
        "negations_found": [tokens[i] for i in neg_idxs],
        "negated_positive_cues": neg_pos_flips,
        "negated_negative_cues": neg_neg_flips,
        "pivot_cues": pivots_found,
        "intensifiers": intens_found,
        "lexical_diversity": round(lex_div, 3),
        "emotion_lex_hits": {emo: round(d * 100.0, 1) for emo, d in emo_densities.items()},
    }


class AffectiveSyntaxExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer extracting 15 non-negative affective, negation-window,
    intensifier, and lexical complexity features as a CSR sparse matrix.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [analyze_syntax_signals(doc)["vector"] for doc in X]
        return csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return np.array(SYNTAX_FEATURE_NAMES)


def build_hybrid_feature_union(
    use_bow_counts=True,
    use_word_ngrams=True,
    ngram_range=(1, 2),
    use_char_ngrams=True,
    use_syntax=True,
    syntax_scaler="maxabs",
):
    """Compose a multi-granularity scaled BoW + Sublinear TF-IDF + Char-WB + Affective Syntax FeatureUnion."""
    transformers = []
    weights = {}
    if use_bow_counts:
        transformers.append((
            "bow",
            Pipeline([
                ("cv", CountVectorizer(ngram_range=ngram_range, min_df=2, max_features=16000)),
                ("scale", MaxAbsScaler()),
            ]),
        ))
        weights["bow"] = 0.9
    if use_word_ngrams:
        transformers.append((
            "word",
            TfidfVectorizer(
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=2,
                max_features=16000,
            ),
        ))
        weights["word"] = 1.0
    if use_char_ngrams:
        transformers.append((
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=6,
                max_features=12000,
            ),
        ))
        weights["char"] = 0.65
    if use_syntax:
        if syntax_scaler == "maxabs":
            syn_step = Pipeline([("extract", AffectiveSyntaxExtractor()), ("scale", MaxAbsScaler())])
        elif syntax_scaler == "std_nomean":
            syn_step = Pipeline([("extract", AffectiveSyntaxExtractor()), ("scale", StandardScaler(with_mean=False))])
        else:
            syn_step = AffectiveSyntaxExtractor()
        transformers.append(("syntax", syn_step))
        weights["syntax"] = 0.45
    return FeatureUnion(transformers, transformer_weights=weights)
