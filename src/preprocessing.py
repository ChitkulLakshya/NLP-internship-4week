"""Text cleaning, tokenization, normalization — no external NLTK data needed."""
import re, html

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_HTML = re.compile(r"<.*?>")
_EMOJI = re.compile("[" + "".join([chr(c) for c in list(range(0x1F600,0x1F650))+list(range(0x1F300,0x1F5FF))+[0x2764,0x263A,0x2665]]) + "]")
_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")

# minimal English stopwords (vendored to avoid nltk.download)
_STOP = set("""a an and are as at be because been before being between both but by can cannot could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself like me more most my myself no nor not of off on once only or other ought our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with you your yours yourself yourselves am will just don should now""".split())
_STOP |= {"rt","amp","via","u"}

def detect_noise(text: str) -> dict:
    t = text or ""
    return {
        "has_url": bool(_URL.search(t)),
        "has_mention": bool(_MENTION.search(t)),
        "has_hashtag": bool(_HASHTAG.search(t)),
        "has_html": bool(_HTML.search(t) or "&amp;" in t or "&lt;" in t),
        "has_emoji": bool(_EMOJI.search(t)),
        "has_repeat": bool(_REPEAT.search(t)),
        "is_missing": (t.strip() == ""),
    }

def clean_text(text: str, lowercase=True, strip_urls=True, strip_mentions=True,
               strip_html=True, keep_hashtag_word=True, collapse_ws=True) -> str:
    if text is None:
        return ""
    t = html.unescape(str(text))
    if strip_html:
        t = _HTML.sub(" ", t)
    if strip_urls:
        t = _URL.sub(" ", t)
    if strip_mentions:
        t = _MENTION.sub(" ", t)
    if keep_hashtag_word:
        t = _HASHTAG.sub(r"\1", t)
    # keep emoji as token <EMOJI> instead of dropping silently
    t = _EMOJI.sub(" <EMOJI> ", t)
    t = re.sub(r"[^A-Za-z0-9<>'$.,!?;:\s-]", " ", t)
    if lowercase:
        t = t.lower()
    t = _REPEAT.sub(r"\1\1", t)  # coooool -> cool (normalize elongation)
    if collapse_ws:
        t = _WS.sub(" ", t).strip()
    return t

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|<emoji>")

def tokenize(text: str, remove_stopwords=False, min_len: int = 2) -> list:
    toks = _TOKEN.findall((text or "").lower())
    out = []
    for w in toks:
        if len(w) < min_len:
            continue
        if remove_stopwords and w in _STOP:
            continue
        if w.isdigit():
            continue
        out.append(w)
    return out

_STEM_SUFFIX = ("ing","ed","es","s","ly","tion","ness","ment")

def simple_stem(w: str) -> str:
    for s in sorted(_STEM_SUFFIX, key=len, reverse=True):
        if w.endswith(s) and len(w)-len(s) >= 3:
            return w[:len(w)-len(s)]
    return w

def normalize_tokens(tokens: list, do_stem=False) -> list:
    if do_stem:
        return [simple_stem(t) for t in tokens]
    return tokens

def preprocess(text: str, remove_stopwords=True, do_stem=False) -> dict:
    cleaned = clean_text(text)
    toks = tokenize(cleaned, remove_stopwords=remove_stopwords)
    normed = normalize_tokens(toks, do_stem=do_stem)
    return {"cleaned": cleaned, "tokens": toks, "normalized": normed, "cleaned_joined": " ".join(normed)}
