"""Week 2 — Text classification: TF-IDF + NB / LinearSVC / LogReg with CV + held-out test."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from src.preprocessing import preprocess
from src.utils import savefig

def build_texts(df, col="text"):
    return df[col].fillna("").astype(str).apply(lambda t: preprocess(t, remove_stopwords=True)["cleaned_joined"]).tolist()

def run_classification(csv_path: str, figdir: str, text_col="text", label_col="label"):
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(figdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    X_raw = df[text_col].fillna("").astype(str).tolist()
    X = [preprocess(t, remove_stopwords=True)["cleaned_joined"] for t in X_raw]
    y = df[label_col].astype(str).tolist()
    labels = sorted(set(y))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "MultinomialNB": make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1,2)), MultinomialNB()),
        "LinearSVC": make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1,2)), LinearSVC()),
        "LogReg": make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1,2)), LogisticRegression(max_iter=1000)),
    }
    results, figs = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in models.items():
        cvres = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["accuracy","f1_macro"])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {
            "cv_acc_mean": float(cvres["test_accuracy"].mean()),
            "cv_acc_std": float(cvres["test_accuracy"].std()),
            "cv_f1_mean": float(cvres["test_f1_macro"].mean()),
            "cv_f1_std": float(cvres["test_f1_macro"].std()),
            "test_acc": float(accuracy_score(yte, pred)),
            "test_f1_macro": float(f1_score(yte, pred, average="macro")),
            "report": classification_report(yte, pred, output_dict=True),
        }
        cm = confusion_matrix(yte, pred, labels=labels)
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Oranges")
        plt.xlabel("pred"); plt.ylabel("true"); plt.title(f"Week2 — Confusion matrix ({name})")
        figs[name] = savefig(os.path.join(figdir, f"w2_cm_{name}.png"))
    # class balance fig
    import pandas as _pd
    plt.figure(figsize=(6,4)); sns.countplot(y=df[label_col], order=labels, color="#e8630a")
    plt.title("Week2 — Class balance"); plt.xticks(rotation=15)
    figs["balance"] = savefig(os.path.join(figdir, "w2_balance.png"))
    # f1 comparison
    plt.figure(figsize=(6,4))
    names = list(results.keys()); f1s = [results[n]["test_f1_macro"] for n in names]
    plt.bar(names, f1s, color=["#1f77b4","#e8630a","#2ca02c"]); plt.ylim(0.5,1.0); plt.ylabel("F1-macro (test)")
    plt.title("Week2 — Model comparison (F1-macro)")
    figs["compare"] = savefig(os.path.join(figdir, "w2_compare.png"))
    summary = {"labels": labels, "models": results, "n_train": len(Xtr), "n_test": len(Xte)}
    return summary, figs
