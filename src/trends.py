"""Week 4 — Topic modeling (LDA), clustering (KMeans), trend detection over time."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_trends(csv_path: str, figdir: str, n_topics: int = 6):
    os.makedirs(figdir, exist_ok=True)
    from src.utils import savefig
    from src.preprocessing import preprocess as pp
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["text"] = df["text"].fillna("").astype(str)
    df["clean"] = df["text"].apply(lambda t: pp(t, remove_stopwords=True)["cleaned_joined"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    # TF-IDF + Count
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df["clean"])
    count = CountVectorizer(max_features=4000, ngram_range=(1,2))
    X_count = count.fit_transform(df["clean"])
    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method="batch")
    lda.fit(X_count)
    terms = count.get_feature_names_out()
    topics = []
    for k in range(n_topics):
        idx = lda.components_[k].argsort()[::-1][:10]
        topics.append([terms[i] for i in idx])
    # KMeans
    km = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    clusters = km.fit_predict(X_tfidf)
    df["cluster"] = clusters
    try:
        sil = float(silhouette_score(X_tfidf, clusters, sample_size=min(2000, len(df))))
    except Exception:
        sil = 0.0
    # Trend: top keywords over months
    track = ["ai", "covid", "election", "cricket", "movie", "price", "climate", "startup"]
    months = sorted(df["month"].unique())
    import re
    trend = {}
    for kw in track:
        vals = []
        for m in months:
            sub = df[df["month"] == m]
            c = sub["clean"].str.contains(rf"\b{kw}\b", regex=True).mean()
            vals.append(float(c))
        trend[kw] = vals
    # figs
    plt.figure(figsize=(9,5))
    for kw, vals in trend.items():
        plt.plot(months, vals, marker="o", label=kw)
    plt.xticks(rotation=45, fontsize=7); plt.ylabel("share of docs"); plt.title("Week4 — Keyword trends over time (monthly share)")
    plt.legend(ncol=2, fontsize=8)
    fig_trend = savefig(os.path.join(figdir, "w4_trends.png"))
    # cluster sizes
    plt.figure(figsize=(7,4)); sns.countplot(x=df["cluster"], color="#e8630a")
    plt.title("Week4 — Cluster sizes (KMeans k=%d)" % n_topics); plt.xlabel("cluster")
    fig_cl = savefig(os.path.join(figdir, "w4_clusters.png"))
    # topic words heatmap-ish bar: first topic
    plt.figure(figsize=(7,4))
    plt.barh(topics[0][::-1], range(10,0,-1), color="#1f77b4"); plt.title("Week4 — Topic-0 top terms (illustrative weights)")
    fig_t0 = savefig(os.path.join(figdir, "w4_topic0.png"))
    # docs per month
    cntm = df["month"].value_counts().sort_index()
    plt.figure(figsize=(9,4)); plt.bar(cntm.index.tolist(), cntm.values.tolist(), color="#2ca02c")
    plt.xticks(rotation=60, fontsize=7); plt.title("Week4 — Docs per month")
    fig_vol = savefig(os.path.join(figdir, "w4_volume.png"))
    stats = {"n_docs": len(df), "months": months, "n_topics": n_topics, "topics": topics,
             "silhouette": sil, "trend": trend, "cluster_sizes": pd.Series(clusters).value_counts().to_dict()}
    figs = {"trends": fig_trend, "clusters": fig_cl, "topic0": fig_t0, "volume": fig_vol}
    return stats, figs, df
