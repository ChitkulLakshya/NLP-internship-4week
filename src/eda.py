"""Week 1 — EDA: distributions, term/n-gram frequencies, sentiment proxy, noise audit."""
import os, re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .preprocessing import clean_text, tokenize, detect_noise
from .utils import ensure_dir, savefig

POS_LEX = set("love great awesome amazing best excellent fantastic wonderful perfect liked good happy win nice brilliant outstanding".split())
NEG_LEX = set("hate terrible awful worst bad poor boring disappointing horrible waste sad angry fail useless".split())

def lex_sentiment(text: str) -> str:
    toks = tokenize(clean_text(text))
    p = sum(1 for t in toks if t in POS_LEX)
    n = sum(1 for t in toks if t in NEG_LEX)
    if p > n: return "positive"
    if n > p: return "negative"
    return "neutral"

def run_eda_week1(csv_path: str, figdir: str):
    ensure_dir(figdir)
    df = pd.read_csv(csv_path)
    stats = {}
    stats["n_rows"] = int(len(df))
    stats["columns"] = list(df.columns)
    stats["missing_by_col"] = {c: int(pd.isna(df[c]).sum()) for c in df.columns}
    # text length
    df["_raw"] = df["text"].fillna("").astype(str)
    df["_clean"] = df["_raw"].apply(clean_text)
    df["_tokens"] = df["_clean"].apply(lambda t: tokenize(t))
    df["_len_chars"] = df["_raw"].str.len()
    df["_len_tokens"] = df["_tokens"].apply(len)
    stats["len_chars_describe"] = df["_len_chars"].describe().to_dict()
    stats["len_tokens_describe"] = df["_len_tokens"].describe().to_dict()
    if "source" in df.columns:
        stats["source_counts"] = df["source"].value_counts().to_dict()
    if "rating" in df.columns:
        stats["rating_counts"] = df["rating"].value_counts(dropna=False).to_dict()
    # noise audit
    flags = df["_raw"].apply(detect_noise)
    for k in ["has_url","has_mention","has_hashtag","has_html","has_emoji","has_repeat","is_missing"]:
        stats[f"noise_{k}"] = int(sum(1 for f in flags if f[k]))
    stats["noise_any"] = int(sum(1 for f in flags if any(f[k] for k in ["has_url","has_mention","has_hashtag","has_html","has_emoji"])))
    # term freq
    cnt = Counter()
    for toks in df["_tokens"]:
        cnt.update(toks)
    # drop vendored stopwords for top terms readability
    from .preprocessing import _STOP
    top = [(w,c) for w,c in cnt.most_common(60) if w not in _STOP][:30]
    stats["top_terms"] = top
    # bigrams / trigrams
    def ngrams(n):
        c = Counter()
        for toks in df["_tokens"]:
            toks2 = [t for t in toks if t not in _STOP]
            for i in range(len(toks2)-n+1):
                c[" ".join(toks2[i:i+n])] += 1
        return c.most_common(20)
    stats["top_bigrams"] = ngrams(2)
    stats["top_trigrams"] = ngrams(3)
    # sentiment proxy
    df["_sent"] = df["_raw"].apply(lex_sentiment)
    stats["sentiment_counts"] = df["_sent"].value_counts().to_dict()

    figs = {}
    # 1 length hist
    plt.figure(figsize=(7,4)); sns.histplot(df["_len_tokens"], bins=40, kde=True, color="#e8630a")
    plt.xlabel("tokens per doc"); plt.ylabel("count"); plt.title("Week1 — Document length (tokens)")
    figs["len_hist"] = savefig(os.path.join(figdir,"w1_len_hist.png"))
    # 2 source bar
    if "source" in df.columns:
        plt.figure(figsize=(6,4)); sns.countplot(data=df, x="source", order=sorted(df["source"].dropna().unique()), color="#e8630a")
        plt.title("Docs by source"); plt.xticks(rotation=15)
        figs["source"] = savefig(os.path.join(figdir,"w1_source.png"))
    # 3 rating bar
    if "rating" in df.columns:
        plt.figure(figsize=(6,4)); sns.countplot(data=df, x="rating", color="#1f77b4")
        plt.title("Rating distribution")
        figs["rating"] = savefig(os.path.join(figdir,"w1_rating.png"))
    # 4 top terms
    if top:
        w, c = zip(*top[::-1])
        plt.figure(figsize=(7,5)); plt.barh(list(w), list(c), color="#2ca02c")
        plt.title("Top 30 terms (stopwords removed)"); plt.xlabel("frequency")
        figs["top_terms"] = savefig(os.path.join(figdir,"w1_top_terms.png"))
    # 5 bigrams
    if stats["top_bigrams"]:
        w, c = zip(*stats["top_bigrams"][::-1])
        plt.figure(figsize=(7,5)); plt.barh(list(w), list(c), color="#9467bd")
        plt.title("Top 20 bigrams"); 
        figs["bigrams"] = savefig(os.path.join(figdir,"w1_bigrams.png"))
    # 6 sentiment
    plt.figure(figsize=(6,4))
    s = df["_sent"].value_counts()
    plt.bar(s.index.tolist(), s.values.tolist(), color=["#2ca02c","#d62728","#7f7f7f"])
    plt.title("Sentiment proxy (lexicon) distribution")
    figs["sentiment"] = savefig(os.path.join(figdir,"w1_sentiment.png"))
    # 7 noise
    labels = ["URL","mention","hashtag","HTML","emoji","repeat"]
    vals = [stats["noise_has_url"],stats["noise_has_mention"],stats["noise_has_hashtag"],stats["noise_has_html"],stats["noise_has_emoji"],stats["noise_has_repeat"]]
    plt.figure(figsize=(7,4)); plt.bar(labels, vals, color="#ff7f0e"); plt.title("Noisy-pattern counts")
    figs["noise"] = savefig(os.path.join(figdir,"w1_noise.png"))
    # 8 boxplot length by source
    if "source" in df.columns:
        plt.figure(figsize=(7,4)); sns.boxplot(data=df, x="source", y="_len_tokens")
        plt.title("Length by source")
        figs["len_by_source"] = savefig(os.path.join(figdir,"w1_len_by_source.png"))
    return stats, figs, df
