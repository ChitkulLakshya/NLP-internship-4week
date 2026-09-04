"""Week 3 — Sentiment (lexicon + LogReg) and NER (rule-based statistical)."""
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

POS = set("love great awesome amazing best excellent fantastic wonderful perfect liked good happy win nice brilliant outstanding recommend loved great excellent".split())
NEG = set("hate terrible awful worst bad poor boring disappointing horrible waste sad angry fail useless regret hated worst".split())

def lex_predict(text: str) -> str:
    toks = re.findall(r"[a-z']+", (text or "").lower())
    p = sum(1 for t in toks if t in POS)
    n = sum(1 for t in toks if t in NEG)
    if p > n: return "positive"
    if n > p: return "negative"
    return "neutral"

# ---- Rule-based NER ----
P_PERSON = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
P_ORG = re.compile(r"\b([A-Z][A-Za-z&]* (?:Inc|Corp|Labs|Technologies|Studio|Systems|Group|AI|Foods|Airlines|Studios))\b")
P_LOC = re.compile(r"\b(Hyderabad|Mumbai|Delhi|Bangalore|Chennai|New York|London|Paris|Tokyo|San Francisco|Berlin)\b")
P_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b")
P_MONEY = re.compile(r"(\$ ?\d[\d,]*(?:\.\d+)?|\b\d+ (?:dollars|rupees|USD|INR)\b)", re.IGNORECASE)

ENTITY_PATTERNS = [("PERSON", P_PERSON), ("ORG", P_ORG), ("LOC", P_LOC), ("DATE", P_DATE), ("MONEY", P_MONEY)]

def extract_entities(text: str):
    t = text or ""
    ents = []
    for label, pat in ENTITY_PATTERNS:
        for m in pat.finditer(t):
            ents.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end()})
    # de-dup overlapping: prefer longer
    ents = sorted(ents, key=lambda e: (e["start"], -(e["end"]-e["start"])))
    filt = []
    occupied = []
    for e in ents:
        if any(not (e["end"] <= s or e["start"] >= en) for s, en in occupied):
            continue
        filt.append(e); occupied.append((e["start"], e["end"]))
    return filt

def score_ner(pred, gold):
    """Exact-match micro scores on (label, text-lower) sets."""
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        ps = set((e["label"], e["text"].lower()) for e in p)
        gs = set((e["label"], e["text"].lower()) for e in g)
        tp += len(ps & gs); fp += len(ps - gs); fn += len(gs - ps)
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def parse_gold(s: str):
    import ast
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def run_week3(csv_path: str, figdir: str):
    os.makedirs(figdir, exist_ok=True)
    from src.utils import savefig
    from src.preprocessing import preprocess as pp
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    # --- sentiment ---
    df["lex_pred"] = df["text"].apply(lex_predict)
    if "sentiment" in df.columns:
        y = df["sentiment"].astype(str)
        lex_acc = float(accuracy_score(y, df["lex_pred"]))
        lex_f1 = float(f1_score(y, df["lex_pred"], average="macro", zero_division=0))
    else:
        lex_acc = lex_f1 = 0.0
    # ML sentiment (binary-ish; drop neutral for cleaner demo, but report 3-class too)
    X = [pp(t, remove_stopwords=True)["cleaned_joined"] for t in df["text"].tolist()]
    ml = {}
    if "sentiment" in df.columns:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        from sklearn.pipeline import make_pipeline
        pipe = make_pipeline(TfidfVectorizer(max_features=4000, ngram_range=(1,2)), LogisticRegression(max_iter=1000))
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        ml = {"acc": float(accuracy_score(yte, pred)), "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
              "report": classification_report(yte, pred, output_dict=True, zero_division=0)}
        cm = confusion_matrix(yte, pred, labels=sorted(y.unique()))
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), cmap="Greens")
        plt.xlabel("pred"); plt.ylabel("true"); plt.title("Week3 — Sentiment ML confusion matrix")
        fig_cm = savefig(os.path.join(figdir, "w3_sent_cm.png"))
    else:
        fig_cm = None
    plt.figure(figsize=(6,4)); sns.countplot(y=df["lex_pred"], order=["positive","neutral","negative"], color="#e8630a")
    plt.title("Week3 — Lexicon sentiment distribution")
    fig_lex = savefig(os.path.join(figdir, "w3_lex_dist.png"))
    # --- NER ---
    df["pred_ents"] = df["text"].apply(extract_entities)
    gold = df["entities"].apply(parse_gold).tolist() if "entities" in df.columns else [[] for _ in range(len(df))]
    ner_scores = score_ner(df["pred_ents"].tolist(), gold)
    lab_c = Counter(e["label"] for lst in df["pred_ents"].tolist() for e in lst)
    plt.figure(figsize=(6,4)); plt.bar(list(lab_c.keys()), list(lab_c.values()), color="#9467bd")
    plt.title("Week3 — Predicted entity-type histogram")
    fig_ner = savefig(os.path.join(figdir, "w3_ner_hist.png"))
    # per-type recall proxy (entity present in gold)
    stats = {"lex_acc": lex_acc, "lex_f1": lex_f1, "ml": ml, "ner": ner_scores,
             "n_docs": len(df), "pred_type_counts": dict(lab_c)}
    figs = {"lex_dist": fig_lex, "ner_hist": fig_ner}
    if fig_cm: figs["sent_cm"] = fig_cm
    return stats, figs, df
