"""
core/live_pulse_service.py
--------------------------
Real-world, everyday consumer NLP services powered by VibeCheck's 90.46% Hybrid Ensemble:
  1. Movie & Show Audience Vibe Scanner (Wikipedia Critical Reception + Google News RSS + TVMaze)
  2. Live World & Topic Mood Pulse (Google News RSS + Hacker News stories)
  3. Before-You-Send Message & Email Tone Guard (trigger detection + natural rewrites re-scored live)
"""

from collections import Counter
from html import unescape
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
import requests

from core.llm_client import call_llm_raw, parse_json_envelope, resolve_llm_credentials
from core.nlp_service import EMOTION_META, EmotionNLPService


USER_AGENT = "VibeCheckAI/4.0 (educational-affective-nlp; https://github.com/syed22star-dot/emotisense-ai)"
_PULSE_CACHE = {}
_CACHE_TTL_SEC = 900.0


def _strip_html(raw: str) -> str:
    txt = re.sub(r"<sup[^>]*>.*?</sup>", " ", raw or "", flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<style[^>]*>.*?</style>", " ", txt, flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = unescape(txt)
    txt = re.sub(r"\[\d+\]|\[edit\]|\[citation needed\]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _split_clean_sentences(text: str, max_items: int = 18) -> list[str]:
    out = []
    seen = set()
    for part in re.split(r"(?<=[.!?])\s+", text or ""):
        s = part.strip(" -–—\"'\t\r\n")
        words = re.findall(r"[A-Za-z]+", s)
        if len(s) < 35 or len(s) > 260 or len(words) < 6:
            continue
        low = s.lower()
        if any(bad in low for bad in ("archived from the original", "retrieved ", "isbn ", "doi:", "box office gross", "rotten tomatoes gives")):
            if not any(tone in low for tone in ("praised", "loved", "thrilling", "emotional", "moving", "disappointing", "scary", "funny", "critics", "acclaimed")):
                continue
        key = re.sub(r"\W+", "", low)[:64]
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= max_items:
            break
    return out


class LivePulseService:
    """Fetches real public reaction snippets and scores them with VibeCheck's trained Hybrid NLP models."""

    def __init__(self, nlp_service: EmotionNLPService):
        self.nlp = nlp_service
        self.headers = {"User-Agent": USER_AGENT, "Accept": "application/json, application/xml, text/xml"}

    def _get_cached(self, key: str):
        hit = _PULSE_CACHE.get(key)
        if hit and (time.time() - hit["ts"]) < _CACHE_TTL_SEC:
            return hit["data"]
        return None

    def _set_cache(self, key: str, data: dict):
        _PULSE_CACHE[key] = {"ts": time.time(), "data": data}

    def fetch_google_news_rss(self, query: str, limit: int = 16) -> list[dict]:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        })
        items = []
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(3, 6))
            if not resp.ok:
                return []
            root = ET.fromstring(resp.text)
            seen = set()
            for item in root.findall("./channel/item"):
                raw_title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                src_el = item.find("source")
                source_name = (src_el.text.strip() if src_el is not None and src_el.text else "")
                if " - " in raw_title:
                    headline, tail = raw_title.rsplit(" - ", 1)
                    if not source_name:
                        source_name = tail.strip()
                else:
                    headline = raw_title
                clean_h = _strip_html(headline)
                if len(re.findall(r"[A-Za-z]+", clean_h)) < 4:
                    continue
                k = clean_h.lower()[:70]
                if k in seen:
                    continue
                seen.add(k)
                items.append({
                    "text": clean_h,
                    "source": f"News · {source_name}" if source_name else "Google News",
                    "url": link,
                    "date": pub[:16],
                })
                if len(items) >= limit:
                    break
        except Exception:
            return []
        return items

    def fetch_wikipedia_reception(self, title: str, max_quotes: int = 16) -> dict:
        out = {"matched_title": None, "summary": "", "quotes": [], "raw_reception_text": "", "categories_text": ""}
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": f"{title} film movie", "srlimit": 8, "format": "json"},
                headers=self.headers,
                timeout=(3, 6),
            )
            if not search_resp.ok:
                return out
            hits = (search_resp.json().get("query") or {}).get("search") or []
            if not hits:
                return out

            q_words = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in ("the", "a", "an", "movie", "film")]
            scored_hits = []
            for idx, h in enumerate(hits):
                ht = h.get("title", "")
                hl = ht.lower()
                snip = _strip_html(h.get("snippet", "")).lower()
                bonus = 0
                if "(film)" in hl or "film)" in hl or "(tv series)" in hl or "series)" in hl:
                    bonus += 60
                if q_words and all(w in hl for w in q_words):
                    bonus += 35
                if any(k in snip for k in ("directed by", "starring", "feature film", "action drama film", "television series", "motion picture", "box office")):
                    bonus += 40
                if any(k in snip for k in ("is an indian actress", "is an american actor", "politician", "cricketer", "singer-songwriter", "born ")) and "film by" not in snip and "directed by" not in snip:
                    bonus -= 55
                if hl.strip() == title.lower().strip():
                    bonus += 20
                scored_hits.append((bonus - idx * 3, ht))
            scored_hits.sort(reverse=True)
            chosen_title = scored_hits[0][1]
            out["matched_title"] = chosen_title

            sec_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "parse", "page": chosen_title, "prop": "sections", "format": "json"},
                headers=self.headers,
                timeout=(3, 6),
            )
            if sec_resp.ok:
                secs = (sec_resp.json().get("parse") or {}).get("sections") or []
                rec_sec = next(
                    (s for s in secs if any(kw in str(s.get("line", "")).lower() for kw in ("critical response", "critical reception", "reception", "reviews", "audience"))),
                    None,
                )
                if rec_sec:
                    html_resp = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={"action": "parse", "page": chosen_title, "section": rec_sec["index"], "prop": "text", "format": "json"},
                        headers=self.headers,
                        timeout=(3, 6),
                    )
                    if html_resp.ok:
                        raw_html = ((html_resp.json().get("parse") or {}).get("text") or {}).get("*", "")
                        plain = _strip_html(raw_html)
                        out["raw_reception_text"] = plain
                        out["quotes"] = _split_clean_sentences(plain, max_items=max_quotes)

            intro_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "titles": chosen_title, "prop": "extracts|categories", "exintro": 1, "explaintext": 1, "cllimit": 20, "format": "json"},
                headers=self.headers,
                timeout=(3, 6),
            )
            if intro_resp.ok:
                pages = (intro_resp.json().get("query") or {}).get("pages") or {}
                first_page = next(iter(pages.values()), {})
                extract = str(first_page.get("extract") or "").strip()
                cats = [str(c.get("title", "")).replace("Category:", "") for c in (first_page.get("categories") or [])]
                out["categories_text"] = " ".join(cats)
                out["summary"] = extract[:360] + ("..." if len(extract) > 360 else "")
                if len(out["quotes"]) < 5 and extract:
                    extra = _split_clean_sentences(extract, max_items=8)
                    existing = {q[:45].lower() for q in out["quotes"]}
                    for q in extra:
                        if q[:45].lower() not in existing:
                            out["quotes"].append(q)
        except Exception:
            return out
        return out

    def fetch_tvmaze_info(self, title: str) -> dict:
        try:
            resp = requests.get(
                "https://api.tvmaze.com/search/shows",
                params={"q": title},
                headers={"User-Agent": USER_AGENT},
                timeout=(2.5, 5),
            )
            if not resp.ok:
                return {}
            rows = resp.json() or []
            if not rows:
                return {}
            show = rows[0].get("show") or {}
            q_low = title.lower().strip()
            s_name = str(show.get("name") or "").lower().strip()
            if q_low not in s_name and s_name not in q_low:
                return {}
            return {
                "name": show.get("name"),
                "genres": show.get("genres") or [],
                "rating": (show.get("rating") or {}).get("average"),
                "premiered": (show.get("premiered") or "")[:4],
                "summary": _strip_html(show.get("summary") or "")[:280],
            }
        except Exception:
            return {}

    def fetch_hn_stories(self, topic: str, limit: int = 10) -> list[dict]:
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": topic, "tags": "story", "hitsPerPage": limit * 2},
                headers={"User-Agent": USER_AGENT},
                timeout=(2.5, 5),
            )
            if not resp.ok:
                return []
            kw_tokens = [w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) >= 2]
            out = []
            for h in resp.json().get("hits") or []:
                t = str(h.get("title") or "").strip()
                if not t or len(re.findall(r"[A-Za-z]+", t)) < 4:
                    continue
                tl = t.lower()
                if kw_tokens and not any(tok in tl for tok in kw_tokens):
                    continue
                out.append({
                    "text": t,
                    "source": "Hacker News",
                    "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    "date": str(h.get("created_at") or "")[:10],
                })
                if len(out) >= limit:
                    break
            return out
        except Exception:
            return []

    def _aggregate_snippets(self, snippets: list[dict], model_key: str) -> dict:
        emotions = list(EMOTION_META.keys())
        prob_sums = {e: 0.0 for e in emotions}
        vote_counts = Counter()
        trig_counter = Counter()
        scored_items = []

        for item in snippets:
            txt = item.get("text", "").strip()
            if not txt:
                continue
            try:
                res = self.nlp.analyze_single(txt, model_key=model_key, compare_all=False)
            except Exception:
                continue
            for e in emotions:
                prob_sums[e] += float(res["all_probabilities"].get(e, 0.0))
            vote_counts[res["emotion"]] += 1
            for tok in res.get("top_positive_words", [])[:3]:
                if len(tok["word"]) >= 3:
                    trig_counter[tok["word"]] += 1
            scored_items.append({
                "text": txt,
                "source": item.get("source", "Live Web"),
                "url": item.get("url", ""),
                "date": item.get("date", ""),
                "emotion": res["emotion"],
                "emoji": res["emoji"],
                "color": res["color"],
                "confidence": res["confidence"],
                "top_word": (res["top_positive_words"][0]["word"] if res.get("top_positive_words") else ""),
            })

        n = len(scored_items)
        if n == 0:
            raise ValueError("Could not gather enough readable reaction quotes for that search. Try a slightly broader movie/show title or topic.")

        mix = {e: round(prob_sums[e] / n, 1) for e in emotions}
        dominant = max(mix.items(), key=lambda x: x[1])[0]
        sorted_mix = sorted(mix.items(), key=lambda x: x[1], reverse=True)
        runner_up = sorted_mix[1][0] if len(sorted_mix) > 1 else dominant

        # Derived playful indices (bounded 0-100)
        hype_joy = round(min(100.0, mix.get("joy", 0.0) + 0.65 * mix.get("love", 0.0)), 1)
        suspense = round(min(100.0, mix.get("fear", 0.0) + 0.75 * mix.get("surprise", 0.0)), 1)
        tearjerker = round(min(100.0, mix.get("sadness", 0.0) * 1.15), 1)
        heat_rant = round(min(100.0, mix.get("anger", 0.0) * 1.15), 1)

        warmth_score = round(
            min(100.0, max(0.0, 50.0 + 0.5 * (mix["joy"] + mix["love"] + 0.4 * mix["surprise"] - mix["anger"] - mix["fear"] - mix["sadness"]))),
            1,
        )
        polarization = round(min(100.0, max(0.0, (1.0 - (sorted_mix[0][1] - sorted_mix[1][1]) / 100.0) * 100.0)), 1)

        return {
            "sample_count": n,
            "dominant_emotion": dominant,
            "dominant_emoji": EMOTION_META[dominant]["emoji"],
            "runner_up_emotion": runner_up,
            "emotion_mix": mix,
            "vote_counts": {e: int(vote_counts.get(e, 0)) for e in emotions},
            "vibe_indices": {
                "hype_joy": hype_joy,
                "suspense": suspense,
                "tearjerker": tearjerker,
                "heat_rant": heat_rant,
                "warmth_temperature": warmth_score,
                "polarization": polarization,
            },
            "top_triggers": [{"word": w, "count": int(c)} for w, c in trig_counter.most_common(12)],
            "items": scored_items,
        }

    def _build_movie_verdict(self, dominant: str, indices: dict, title: str) -> tuple[str, str]:
        h, s, t, r = indices["hype_joy"], indices["suspense"], indices["tearjerker"], indices["heat_rant"]
        if dominant == "joy" or h >= 42:
            return ("Crowd-Pleasing Joy & Excitement", f"Real critic & audience reactions for {title} lean strongly warm, energized, and appreciative.")
        if dominant == "fear" or s >= 34:
            return ("Edge-of-Your-Seat Thriller Vibes", f"Discussions around {title} light up suspense, tension, and high-stakes surprise.")
        if dominant == "sadness" or t >= 30:
            return ("Tissue-Box Emotional Drama", f"Reviews highlight deeply moving, bittersweet, and poignant storytelling in {title}.")
        if dominant == "anger" or r >= 30:
            return ("Fiery / Heated Critic Debate", f"Reactions to {title} carry sharp frustration, clash, or intense dramatic friction.")
        if dominant == "love":
            return ("Pure Comfort & Fan Adoration", f"Audiences show standout affection and heartfelt connection toward {title}.")
        return ("Twist-Filled Wildcard Ride", f"Reactions for {title} blend astonishment, curiosity, and multi-layered twists.")

    def filter_bot_reviews(self, raw_snippets: list[dict]) -> tuple[list[dict], dict]:
        """Remove generic hype/bot promo spam & near-duplicate copy-paste items."""
        cleaned = []
        seen_sets = []
        dropped = 0
        spam_cues = ("download hd", "watch full movie online", "telegram link", "free streaming", "1000cr loading", "buy tickets now", "promo code", "click link bio")
        for item in raw_snippets:
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            words = re.findall(r"[A-Za-z0-9]+", txt.lower())
            letters = [ch for ch in txt if ch.isalpha()]
            upper_ratio = (sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)) if letters else 0.0
            low = txt.lower()
            if len(words) < 5 or (upper_ratio > 0.55 and len(txt) > 18) or any(sp in low for sp in spam_cues):
                dropped += 1
                continue
            wset = set(words)
            is_dup = any((len(wset & prev) / max(len(wset | prev), 1)) >= 0.72 for prev in seen_sets)
            if is_dup:
                dropped += 1
                continue
            seen_sets.append(wset)
            cleaned.append(item)

        raw_n = max(len(raw_snippets), 1)
        clean_pct = round((len(cleaned) / raw_n) * 100, 1)
        return cleaned, {
            "raw_count": len(raw_snippets),
            "verified_count": len(cleaned),
            "filtered_count": dropped,
            "cleanliness_pct": clean_pct,
            "summary": f"{len(cleaned)} editorial & verified items kept · {dropped} low-signal/duplicate blurbs filtered",
        }

    def extract_critic_scores(self, raw_text: str, tv_info: dict) -> dict:
        """Extract verified Rotten Tomatoes %, Metacritic /100, or TVMaze scores cited in real prose."""
        rt_score = None
        mc_score = None
        txt = raw_text or ""
        m_rt1 = re.search(r"Rotten Tomatoes[^.\n]{0,110}?(\d{1,3})\s*%", txt, re.IGNORECASE)
        m_rt2 = re.search(r"(\d{1,3})\s*%[^.\n]{0,70}?Rotten Tomatoes", txt, re.IGNORECASE)
        if m_rt1 and 0 <= int(m_rt1.group(1)) <= 100:
            rt_score = int(m_rt1.group(1))
        elif m_rt2 and 0 <= int(m_rt2.group(1)) <= 100:
            rt_score = int(m_rt2.group(1))

        m_mc1 = re.search(r"Metacritic[^.\n]{0,110}?(\d{1,3})\s*(?:out of 100|\/100)", txt, re.IGNORECASE)
        m_mc2 = re.search(r"(\d{1,3})\s*(?:out of 100|\/100)[^.\n]{0,60}?Metacritic", txt, re.IGNORECASE)
        if m_mc1 and 0 <= int(m_mc1.group(1)) <= 100:
            mc_score = int(m_mc1.group(1))
        elif m_mc2 and 0 <= int(m_mc2.group(1)) <= 100:
            mc_score = int(m_mc2.group(1))

        tv_rating = tv_info.get("rating") if isinstance(tv_info, dict) else None
        signals = [v for v in (rt_score, mc_score, (float(tv_rating) * 10.0 if tv_rating else None)) if v is not None]
        combined_critic = round(sum(signals) / len(signals), 1) if signals else None
        return {
            "rotten_tomatoes": rt_score,
            "metacritic": mc_score,
            "tvmaze_rating": tv_rating,
            "combined_critic_pct": combined_critic,
        }

    def classify_audience_rating(self, title: str, corpus: str, categories_text: str, tv_info: dict) -> dict:
        """Separate clear Family Entertainer vs Teen/UA vs Adult-Rated indicator from real metadata/reception."""
        text = f"{title} {categories_text} {corpus} {' '.join((tv_info or {}).get('genres', []))}".lower()
        adult_cues = ["rated r", "nc-17", "tv-ma", "18+", "adults only", "certified a", "certificate a", "graphic violence", "explicit sexual", "nudity", "extreme gore", "strong profanity", "erotic"]
        teen_cues = ["pg-13", "tv-14", "certified ua", "certificate ua", "u/a", "action thriller", "action drama", "crime thriller", "bloodshed", "smuggling", "gangster", "gunfire", "intense violence", "martial arts", "horror", "assassination"]
        family_cues = ["family film", "family entertainer", "all ages", "animated", "certified u", "rated g", "rated pg", "tv-g", "tv-pg", "comedy-drama", "heartwarming", "coming-of-age"]

        found_adult = [c for c in adult_cues if re.search(r"\b" + re.escape(c) + r"\b", text)]
        found_teen = [c for c in teen_cues if re.search(r"\b" + re.escape(c) + r"\b", text)]
        found_fam = [c for c in family_cues if re.search(r"\b" + re.escape(c) + r"\b", text)]

        if found_adult:
            return {
                "tier": "adult",
                "label": "Adult-Rated / Mature (18+ / R / A)",
                "short_badge": "Adult-Rated Movie",
                "emoji": "🔞",
                "reason": f"Mature content or A/R signals detected ({', '.join(found_adult[:2])}) — best for adults rather than family/kids viewing.",
            }
        if found_teen and not found_fam:
            return {
                "tier": "teen",
                "label": "Teen & Parental Guidance (UA / PG-13)",
                "short_badge": "Parental Guidance (UA / PG-13)",
                "emoji": "⚠️",
                "reason": f"Intense action/crime or PG-13/UA themes noted ({', '.join(found_teen[:2])}) — fine for teens & adults, cautious for young kids.",
            }
        return {
            "tier": "family",
            "label": "Family Entertainer (All-Ages Friendly)",
            "short_badge": "Family Entertainer",
            "emoji": "👨‍👩‍👧‍👦",
            "reason": "Clean comedy, heartwarming drama, or U/PG family signals with no graphic adult triggers found.",
        }

    def compute_watch_verdict_pie(self, agg: dict, critic_scores: dict, raw_corpus: str) -> dict:
        """Compute 4 actionable watch tiers (Definitely Watch, Good to Watch, One-Time Watch, Not That Good)."""
        vi = agg["vibe_indices"]
        mix = agg["emotion_mix"]
        low = (raw_corpus or "").lower()

        praise_words = ("acclaimed", "masterpiece", "brilliant", "entertaining", "delightful", "blockbuster", "heartfelt", "gripping", "superb", "triumph", "hilarious", "memorable", "praised", "loved", "unforgettable")
        neg_words = ("disappointing", "weak", "boring", "sluggish", "messy", "bloated", "tedious", "cliche", "predictable", "forgettable", "poorly", "underwhelming", "flawed", "terrible")
        praise_hits = sum(low.count(w) for w in praise_words)
        neg_hits = sum(low.count(w) for w in neg_words)
        editorial_bonus = (praise_hits - 1.4 * neg_hits) * 4.5

        base_quality = 46.0 + 0.55 * vi["hype_joy"] + 0.25 * vi["suspense"] + 0.15 * vi["tearjerker"] - 0.55 * vi["heat_rant"] + editorial_bonus
        if critic_scores.get("combined_critic_pct") is not None:
            base_quality = 0.55 * base_quality + 0.45 * float(critic_scores["combined_critic_pct"])
        q_score = max(12.0, min(97.0, base_quality))

        # Distance-based weights over 4 tiers centered at 85, 68, 50, 26
        centers = [
            ("definitely", "Definitely Watch", "#059669", 84.0),
            ("good", "Good to Watch", "#7c3aed", 67.0),
            ("onetime", "One-Time Watchable", "#d97706", 48.0),
            ("skip", "Not That Good", "#e11d48", 25.0),
        ]
        raw_w = []
        for k, lbl, col, ctr in centers:
            w = 100.0 / (1.0 + ((q_score - ctr) / 18.0) ** 2)
            raw_w.append((k, lbl, col, w))
        tot_w = sum(x[3] for x in raw_w) or 1.0
        slices = []
        running_sum = 0
        for idx, (k, lbl, col, w) in enumerate(raw_w):
            pct = round((w / tot_w) * 100) if idx < len(raw_w) - 1 else max(0, 100 - running_sum)
            running_sum += pct
            slices.append({"key": k, "label": lbl, "pct": int(pct), "color": col})

        top_slice = max(slices, key=lambda s: s["pct"])
        headlines = {
            "definitely": ("Definitely Watch — Go For It!", "Strong critic praise & warm audience vibe make this an easy recommend."),
            "good": ("Good to Watch — Worth Your Time", "Positive reviews and solid entertainment appeal outweigh minor complaints."),
            "onetime": ("One-Time Watchable — Fun Weekend Pick", "Divided or mixed reactions: enjoyable once with popcorn, not essential cinema."),
            "skip": ("Not That Good — Wait or Skip", "Critics and discussions flag noticeable flaws, drag, or frustration."),
        }
        hl, rsn = headlines[top_slice["key"]]
        return {
            "quality_index": round(q_score, 1),
            "winning_key": top_slice["key"],
            "winning_label": top_slice["label"],
            "headline": hl,
            "reason": rsn,
            "slices": slices,
        }

    def analyze_movie_vibe(self, title: str, model_key: str = "hybrid_ensemble") -> dict:
        clean_title = (title or "").strip()
        if len(clean_title) < 2:
            raise ValueError("Enter a movie or show name (e.g., Inception, 3 Idiots, Stranger Things).")

        chosen = model_key if model_key in self.nlp.loaded_models else self.nlp.get_primary_key()
        cache_key = f"movie_v2:{chosen}:{clean_title.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return {**cached, "cached": True}

        wiki = self.fetch_wikipedia_reception(clean_title, max_quotes=16)
        news_items = self.fetch_google_news_rss(f'"{clean_title}" (movie OR film OR series) (review OR audience OR critics)', limit=14)
        tv_info = self.fetch_tvmaze_info(clean_title)

        snippets_raw = []
        for q in wiki.get("quotes", []):
            snippets_raw.append({"text": q, "source": f"Wikipedia · {wiki.get('matched_title') or clean_title}", "url": "", "date": "Critics/Reception"})
        snippets_raw.extend(news_items)
        if tv_info.get("summary") and len(snippets_raw) < 6:
            snippets_raw.append({"text": tv_info["summary"], "source": "TVMaze Synopsis", "url": "", "date": tv_info.get("premiered", "")})

        snippets_clean, bot_filter = self.filter_bot_reviews(snippets_raw)
        if not snippets_clean and snippets_raw:
            snippets_clean = snippets_raw

        agg = self._aggregate_snippets(snippets_clean[:28], chosen)
        full_corpus = " ".join([
            wiki.get("raw_reception_text", ""),
            wiki.get("summary", ""),
            tv_info.get("summary", ""),
            " ".join(s["text"] for s in snippets_clean),
        ])
        critic_scores = self.extract_critic_scores(full_corpus, tv_info)
        audience_suitability = self.classify_audience_rating(clean_title, full_corpus, wiki.get("categories_text", ""), tv_info)
        watch_verdict_pie = self.compute_watch_verdict_pie(agg, critic_scores, full_corpus)
        verdict_title, verdict_sub = self._build_movie_verdict(agg["dominant_emotion"], agg["vibe_indices"], clean_title)

        payload = {
            "kind": "movie",
            "query": clean_title,
            "matched_title": wiki.get("matched_title") or tv_info.get("name") or clean_title,
            "summary": tv_info.get("summary") or wiki.get("summary") or "",
            "meta": {
                "genres": tv_info.get("genres") or [],
                "rating": tv_info.get("rating"),
                "premiered": tv_info.get("premiered") or "",
                "wiki_quotes": len(wiki.get("quotes", [])),
                "news_quotes": len(news_items),
            },
            "critic_scores": critic_scores,
            "bot_filter": bot_filter,
            "audience_suitability": audience_suitability,
            "watch_verdict_pie": watch_verdict_pie,
            "model_used": self.nlp.loaded_models[chosen]["name"],
            "verdict_title": verdict_title,
            "verdict_subtitle": verdict_sub,
            **agg,
            "cached": False,
        }
        self._set_cache(cache_key, payload)
        return payload

    def analyze_topic_pulse(self, topic: str, model_key: str = "hybrid_ensemble") -> dict:
        clean_topic = (topic or "").strip()
        if len(clean_topic) < 2:
            raise ValueError("Enter a live topic (e.g., Cricket World Cup, iPhone, Nvidia, Climate).")

        chosen = model_key if model_key in self.nlp.loaded_models else self.nlp.get_primary_key()
        cache_key = f"topic:{chosen}:{clean_topic.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return {**cached, "cached": True}

        news_items = self.fetch_google_news_rss(clean_topic, limit=18)
        hn_items = self.fetch_hn_stories(clean_topic, limit=8)
        snippets_raw = news_items + hn_items
        snippets, _ = self.filter_bot_reviews(snippets_raw)
        if not snippets and snippets_raw:
            snippets = snippets_raw
        agg = self._aggregate_snippets(snippets[:24], chosen)

        temp = agg["vibe_indices"]["warmth_temperature"]
        pol = agg["vibe_indices"]["polarization"]
        mood_label = "Upbeat & Energized" if temp >= 60 else ("Tense / Critical" if temp <= 42 else "Mixed & Balanced")
        if pol >= 78:
            mood_label += " · High Debate"

        payload = {
            "kind": "topic",
            "query": clean_topic,
            "model_used": self.nlp.loaded_models[chosen]["name"],
            "mood_headline": mood_label,
            "sources_breakdown": {"news": len(news_items), "hn": len(hn_items)},
            **agg,
            "cached": False,
        }
        self._set_cache(cache_key, payload)
        return payload

    def _deterministic_rewrites(self, text: str, goal: str, diag: dict) -> list[dict]:
        cleaned = re.sub(r"!{2,}", ".", text.strip())
        cleaned = re.sub(r"\b[A-Z]{4,}\b", lambda m: m.group(0).lower(), cleaned)
        softeners = [
            (r"\bwhy did nobody\b", "I noticed we haven't yet aligned on why nobody"),
            (r"\byou always\b", "I've noticed this happens sometimes"),
            (r"\byou never\b", "I'd really appreciate more consistent updates when"),
            (r"\bridge?diculous\b|\bridiculous\b|\bpathetic\b|\bstupid\b|\buseless\b", "frustrating"),
            (r"\bi hate\b", "I'm struggling with"),
            (r"\bthis is unacceptable\b", "we need to sort this out soon"),
            (r"\bfurious\b|\bmad\b|\bpissed\b", "concerned"),
            (r"\basap\b|\bright now\b", "as soon as you have a window today"),
        ]
        soft = cleaned
        for pat, rep in softeners:
            soft = re.sub(pat, rep, soft, flags=re.IGNORECASE)
        if soft.endswith(("!", "?")):
            soft = soft[:-1] + "."

        goal_wraps = {
            "warm": (
                f"Hey! Just wanted to share a quick thought — {soft} Really appreciate your help pulling this together.",
                f"Thanks so much for your time on this. {soft} Let me know what works best on your end!",
            ),
            "calm": (
                f"Hi, following up clearly on our next step: {soft} Happy to align on a practical fix today.",
                f"I want to make sure we resolve this smoothly. {soft} Could we confirm the owner and timeline?",
            ),
            "excited": (
                f"Really looking forward to moving forward! {soft} Glad we are making great progress together!",
                f"Love the momentum here — {soft} Excited to share how helpful this will be!",
            ),
            "apology": (
                f"So sorry for the friction earlier — I appreciate your patience. {soft} Here is how I'll fix it.",
                f"I genuinely regret any stress this caused. {soft} Thank you for sticking with me on this.",
            ),
        }
        r1, r2 = goal_wraps.get(goal, goal_wraps["calm"])
        return [
            {"label": "Practical & Polished", "text": r1, "why": "Lowers defensive trigger words while keeping your point crisp."},
            {"label": "Short & Collaborative", "text": r2, "why": "Shifts focus from blame toward an easy mutual next step."},
        ]

    def run_tone_guard(self, draft_text: str, goal: str = "calm", request_headers=None, provider_override: str = "") -> dict:
        text = str(draft_text or "").strip()
        if not text:
            raise ValueError("Paste an email, Slack note, or message first.")

        goal_norm = goal.lower().strip()
        if goal_norm not in ("warm", "calm", "excited", "apology"):
            goal_norm = "calm"

        model_key = self.nlp.get_primary_key()
        orig_res = self.nlp.analyze_single(text, model_key=model_key, compare_all=False)
        top_emo = orig_res["emotion"]
        probs = orig_res["all_probabilities"]
        harsh_risk = round(probs.get("anger", 0.0) + 0.55 * probs.get("fear", 0.0), 1)

        if goal_norm in ("warm", "excited") and top_emo in ("joy", "love"):
            status_level = "great"
            verdict = f"Reads welcoming & clear ({orig_res['confidence']}% {top_emo.title()}) — safe to send!"
        elif top_emo == "anger" or probs.get("anger", 0.0) >= 35.0:
            status_level = "warning"
            verdict = f"Heads up: reads with {probs.get('anger', 0.0):.0f}% Anger/Friction — could feel sharper than intended."
        elif top_emo == "fear" or probs.get("fear", 0.0) >= 38.0:
            status_level = "caution"
            verdict = f"Heads up: carries {probs.get('fear', 0.0):.0f}% Anxiety/Urgency — might sound stressed or tense."
        elif top_emo == "sadness" and goal_norm != "apology":
            status_level = "caution"
            verdict = f"Gentle note: reads somewhat heavy ({probs.get('sadness', 0.0):.0f}% Sadness) for a '{goal_norm}' message."
        else:
            status_level = "good"
            verdict = f"Solid draft ({orig_res['confidence']}% {top_emo.title()}) — minor polish options shown below."

        rewrites_raw = []
        rewrite_source = "Rule-Guided Reframer"
        provider, api_key = resolve_llm_credentials({"provider": provider_override}, request_headers or {})
        if api_key:
            goal_desc = {
                "warm": "warm, kind, everyday friendly",
                "calm": "calm, clear, confident, respectful without sounding stiff",
                "excited": "genuinely upbeat, enthusiastic, grateful",
                "apology": "sincere, accountable, gentle, human",
            }[goal_norm]
            sys_p = (
                "You rewrite real-world messages/emails so they sound completely natural, concise, and written by a real human (never corporate AI clichés).\n"
                "Return ONLY valid JSON: {\"rewrites\": [{\"label\": \"Option A\", \"text\": \"...\", \"why\": \"...\"}, {\"label\": \"Option B\", \"text\": \"...\", \"why\": \"...\"}]}"
            )
            usr_p = f"Goal tone: {goal_desc}\nCurrent detected ML tone: {top_emo} ({orig_res['confidence']}%)\nOriginal draft:\n{text}"
            try:
                raw_txt, used_model, _ = call_llm_raw(provider, api_key, sys_p, usr_p)
                parsed = parse_json_envelope(raw_txt)
                cands = parsed.get("rewrites") or []
                if len(cands) >= 2 and all(isinstance(c, dict) and c.get("text") for c in cands[:2]):
                    rewrites_raw = cands[:2]
                    rewrite_source = f"LLM ({used_model}) + ML Verified"
            except Exception:
                rewrites_raw = []

        if not rewrites_raw:
            rewrites_raw = self._deterministic_rewrites(text, goal_norm, orig_res)

        evaluated_rewrites = []
        for idx, rw in enumerate(rewrites_raw):
            r_text = str(rw.get("text", "")).strip()
            try:
                r_res = self.nlp.analyze_single(r_text, model_key=model_key, compare_all=False)
                anger_before = probs.get("anger", 0.0)
                anger_after = r_res["all_probabilities"].get("anger", 0.0)
                evaluated_rewrites.append({
                    "id": idx + 1,
                    "label": rw.get("label", f"Option {idx + 1}"),
                    "text": r_text,
                    "why": rw.get("why", ""),
                    "emotion": r_res["emotion"],
                    "emoji": r_res["emoji"],
                    "confidence": r_res["confidence"],
                    "all_probabilities": r_res["all_probabilities"],
                    "anger_delta": round(anger_after - anger_before, 1),
                })
            except Exception:
                continue

        return {
            "goal": goal_norm,
            "status_level": status_level,
            "verdict": verdict,
            "harsh_risk": harsh_risk,
            "rewrite_source": rewrite_source,
            "original": {
                "text": text,
                "emotion": top_emo,
                "emoji": orig_res["emoji"],
                "confidence": orig_res["confidence"],
                "all_probabilities": probs,
                "trigger_words": orig_res.get("top_positive_words", [])[:6],
                "syntax_signals": orig_res.get("syntax_signals", {}),
            },
            "rewrites": evaluated_rewrites,
        }
