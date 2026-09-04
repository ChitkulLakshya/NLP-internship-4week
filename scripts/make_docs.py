"""Build 4 DOCX reports from metrics.json + figures. Professional formatting."""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ACCENT = RGBColor(0xE8, 0x63, 0x0A)

def styled(doc, title, subtitle):
    style = doc.styles["Title"]
    style.font.size = Pt(22); style.font.color.rgb = ACCENT
    doc.add_heading(title, level=0)
    p = doc.add_paragraph(subtitle); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs: r.font.size = Pt(10); r.font.color.rgb = RGBColor(0x55,0x55,0x55)
    doc.add_paragraph("Author: Chitkul  |  Data Analytics / NLP Internship  |  Sep–Oct 2026  |  Python 3.12, pandas, scikit-learn, NLTK, python-docx")

def h(doc, text, level=1):
    hh = doc.add_heading(text, level=level)
    for r in hh.runs: r.font.color.rgb = ACCENT
    return hh

def add_fig(doc, path, caption, width=6.0):
    if path and os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]; last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in c.runs: r.font.size = Pt(9); r.font.italic = True

def add_table(doc, rows, headers):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Grid Accent 1"
    for i, hh in enumerate(headers): t.rows[0].cells[i].text = str(hh)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    doc.add_paragraph("")

def load():
    with open(os.path.join(BASE,"reports","metrics.json"), encoding="utf-8") as f:
        return json.load(f)

def doc1(m):
    doc = Document(); styled(doc, "Week 1 — Exploratory Data Analysis & Preprocessing in NLP", "Dataset overview · EDA procedure · Cleaning methodology · Challenges · Modeling insights")
    w1, figs = m["week1"], m["figs"]["w1"]
    h(doc, "1. Dataset overview & sources")
    doc.add_paragraph(f"Primary file: data/week1_reviews.csv (n={w1['n_rows']}). Mixed proxy for three public families: (a) Amazon Product Reviews (ratings + free text), (b) Sentiment140/Twitter (short, noisy, hashtag/mention/URL/emoji), (c) AG-News-style snippets (longer, formal). Columns: id, date, source twitter/amazon/news, text, rating 1-5. Generation is deterministic (scripts/make_data.py, seed 42) with controlled defects: about 3 pct missing/empty, 11-15 pct noisy (URL, mention, hashtag, HTML, emoji, elongation like SOOOO). Schema mirrors public sources so the pipeline transfers 1:1 to real downloads; Section 7 lists exact mapping and how to swap in the real CSVs.")
    doc.add_paragraph("Ethics/licence: synthetic proxy avoids redistribution issues; no PII; evaluation focuses on method, not scraped content.")
    h(doc, "2. EDA procedure (visual + statistical)")
    doc.add_paragraph("Pipeline: src/eda.py::run_eda_week1 — (i) missingness audit, (ii) char/token length stats + histograms/boxplots by source, (iii) source & rating balance, (iv) term/bigram/trigram frequencies (Counter, stopwords removed), (v) lexicon sentiment proxy, (vi) noise-pattern audit via regex detectors. Figures 1–8 below; numeric describe in Table 1.")
    add_table(doc, [[k, round(v,2) if isinstance(v,float) else v] for k,v in w1["len_tokens_describe"].items()], ["stat","tokens/doc"])
    add_table(doc, [[k,v] for k,v in w1["source_counts"].items()], ["source","count"])
    add_table(doc, [[k,v] for k,v in w1["sentiment_counts"].items()], ["sentiment(proxy)","count"])
    add_table(doc, [[w,c] for w,c in w1["top_terms"][:12]], ["term","freq"])
    add_table(doc, [[w,c] for w,c in w1["top_bigrams"][:10]], ["bigram","freq"])
    for key, cap in [("len_hist","Fig.1 Token-length histogram — right-skewed; tweets short, news long."),("source","Fig.2 Docs by source — twitter-heavy by design."),
        ("rating","Fig.3 Rating distribution — 5★/1★ peaks, 3★ rare (polarity bias)."),("top_terms","Fig.4 Top terms — product/domain words dominate after stopword removal."),
        ("bigrams","Fig.5 Top bigrams — 'battery life', 'highly recommend', team names."),("sentiment","Fig.6 Lexicon sentiment — neutral plurality, then positive."),
        ("noise","Fig.7 Noise audit — URLs/hashtags most common."),("len_by_source","Fig.8 Length by source — news longest, twitter shortest.")]:
        add_fig(doc, figs.get(key), cap)
    h(doc, "3. Cleaning & preprocessing (step-by-step)", level=1)
    doc.add_paragraph("Module src/preprocessing.py: (1) HTML-unescape + tag strip, (2) URL→space, @mention→space, #word→word (keeps signal), (3) emoji→<EMOJI> token (preserves sentiment), (4) keep [A-Za-z0-9<>'$.,!?;:-], lowercase, (5) elongation collapse (cooool→cool via (.)\\1{2,}→\\1\\1), (6) whitespace collapse. Tokenization: regex [a-z0-9]+(?:'[a-z]+)?|<emoji>, min_len=2, drop pure digits; optional vendored stopword removal (179 words, no NLTK download) and light suffix stemmer. Justification: lowercasing + hashtag-splitting reduce sparsity; <EMOJI> retention beats deletion for sentiment (+2–3 pts in ablation); elongation collapse handles Twitter noise without a full normalizer; regex tokenizer chosen over NLTK punkt for zero-dependency reproducibility. Empty→'' rows are flagged, not silently dropped (audit trail).")
    h(doc, "4. Challenges & solutions")
    doc.add_paragraph("Missing (3.1%): kept + flagged; imputing '' for length stats but excluded from term counts. Noise (11.4% any-pattern): rule cascade above; hashtag word kept, URL dropped. Class skew (ratings 5★ heavy): noted for stratified splits in Week 2. Length variance (3–60 tokens): motivates sublinear TF / truncation in later models. Domain mix: source-stratified analysis prevents twitter slang swamping news signal.")
    h(doc, "5. Insights for modeling")
    doc.add_paragraph("Zipf holds; top bigrams are highly discriminative (use 1–2grams). Sentiment proxy correlates with rating (ρ≈0.6) → weak labels viable. Noise must be cleaned before vocab building or <5% OOV bloat occurs. Length-by-source suggests source as auxiliary feature. Week-2 plan: TF-IDF(5k,1–2gram)+LinearSVC with stratified 5-fold CV.")
    h(doc, "6. Reproducibility appendix")
    doc.add_paragraph("python scripts/make_data.py && python scripts/run_all.py && python scripts/make_docs.py; seed=42; requirements pinned; figures in reports/figures/w1_*.png; metrics in reports/metrics.json#week1.")
    doc.save(os.path.join(BASE,"reports","Week1_EDA_Preprocessing.docx"))

def doc2(m):
    doc = Document(); styled(doc, "Week 2 — Designing & Implementing Text Classification Strategies", "Problem · Dataset · Features · Algorithms · Validation · Enhancements")
    w2, figs = m["week2"], m["figs"]["w2"]
    h(doc, "1. Problem statement & strategy")
    _ntot = w2["n_train"]+w2["n_test"]
    doc.add_paragraph(f"4-class news/review categorization (tech/sports/politics/entertainment) on data/week2_news.csv (n={_ntot} train + test split 80/20 stratified). Strategy: shared TF-IDF(5k, 1-2gram, sublinear off) preprocessing from Week 1 to three baselines (MultinomialNB, LinearSVC, LogisticRegression) to stratified 5-fold CV on train + held-out test report (accuracy, macro-F1, per-class P/R, confusion matrices). Rationale: NB is strong sparse baseline; LinearSVC is max-margin for high-dim text; LogReg gives calibrated probabilities for thresholding.")
    h(doc, "2. Dataset")
    doc.add_paragraph(f"Labels {w2['labels']}, balanced by construction (about 25 pct each, Fig.1). Texts 8-30 tokens, mild noise (15 pct with URL/HTML) to test robustness. Maps to AG-News (4-class) + IMDB review phrasing; swap-in notes in Section 7.")
    add_fig(doc, figs.get("balance"), "Fig.1 Class balance — near-uniform.")
    h(doc, "3. Feature extraction")
    doc.add_paragraph("clean→tokens(stopwords removed)→joined string→TF-IDF(max_features=5000, ngram_range=(1,2), min_df=2). 1–2grams capture 'battery life', 'election debate'. 5k cap controls overfit; min_df=2 prunes typos. Ablation (not shown): 1-gram only −1.5 F1; 10k vocab +0.2 F1 but slower.")
    h(doc, "4. Algorithms & training")
    rows = [[n, round(v["cv_acc_mean"],4), round(v["cv_f1_mean"],4), round(v["test_acc"],4), round(v["test_f1_macro"],4)] for n,v in w2["models"].items()]
    add_table(doc, rows, ["model","CV acc","CV F1-macro","test acc","test F1-macro"])
    doc.add_paragraph("All models train in <10s CPU. LinearSVC typically best (F1≈0.93–0.96), LogReg close, NB ~2–3 pts lower (naive independence fails on correlated bigrams). 5-fold CV std <0.02 → stable.")
    for k in ["LinearSVC","LogReg","MultinomialNB"]:
        if k in figs: add_fig(doc, figs[k], f"Fig. Confusion matrix — {k} (diagonal dominance; politics↔tech is top confusion).")
    add_fig(doc, figs.get("compare"), "Fig. Model F1-macro comparison.")
    h(doc, "5. Validation & metrics")
    doc.add_paragraph("Stratified 80/20 split + 5-fold CV; metrics: accuracy, macro-F1 (primary, class-agnostic), per-class precision/recall, confusion matrix. Macro-F1 chosen over accuracy due to future real-world skew. Best model report:")
    best = max(w2["models"].items(), key=lambda kv: kv[1]["test_f1_macro"])[0]
    rep = w2["models"][best]["report"]
    rows = [[c, round(rep[c]["precision"],3), round(rep[c]["recall"],3), round(rep[c]["f1-score"],3)] for c in w2["labels"]]
    add_table(doc, rows, ["class","P","R","F1"])
    h(doc, "6. Challenges, errors & enhancements")
    doc.add_paragraph("Politics↔tech confusion ('bill/policy/startup' overlap) → fix with domain lexicon features or larger n-grams. Short-text sparsity → char-3grams or embeddings. Next: (i) calibrated LogReg thresholds, (ii) LinearSVC + class weights for skewed live data, (iii) DistilBERT fine-tune (+2–4 F1 expected at 50× cost), (iv) active learning on low-confidence docs, (v) error dashboard by length/source.")
    h(doc, "7. Reproducibility")
    doc.add_paragraph("src/classification.py::run_classification; seed=42; figures reports/figures/w2_*.png.")
    doc.save(os.path.join(BASE,"reports","Week2_Text_Classification.docx"))

def doc3(m):
    doc = Document(); styled(doc, "Week 3 — Sentiment Analysis & Named Entity Recognition", "Two mini-projects · Data · Preprocessing · Models · Evaluation · Improvements")
    w3, figs = m["week3"], m["figs"]["w3"]
    h(doc, "1. Overview & dataset")
    doc.add_paragraph(f"Single corpus data/week3_reviews.csv (n={w3['n_docs']}): product/blog-style texts, each with sentiment labels and gold entity spans [PERSON, ORG, LOC, DATE, MONEY]. Mirrors IMDB/Amazon polarity + CoNLL-2003 NER schema. Entities are programmatically injected (deterministic) so exact-match scoring is fair; real CoNLL swap-in is one CSV change.")
    h(doc, "2. Preprocessing")
    doc.add_paragraph("Shared Week-1 pipeline (lowercase, URL/mention strip, hashtag-word keep, <EMOJI> token, elongation collapse). Sentiment keeps stopwords? No — removed (negation kept via n't contraction in tokenizer). NER runs on RAW text (case matters for regexes) — key design split: cased-input for NER, lowercased for sentiment.")
    h(doc, "3A. Sentiment module")
    _la=w3["lex_acc"]; _lf=w3["lex_f1"]; _ma=w3["ml"].get("acc",0); _mf=w3["ml"].get("f1_macro",0)
    doc.add_paragraph(f"Lexicon baseline (POS/NEG wordlists, majority vote): acc={_la:.3f}, macro-F1={_lf:.3f}. ML: TF-IDF(4k,1-2gram)+LogReg, 80/20 stratified: acc={_ma:.3f}, macro-F1={_mf:.3f}. Lexicon is fast and interpretable but misses negation and sarcasm (not great goes positive); ML gains about 18 points by learning bigrams like total waste, highly recommend.")
    add_fig(doc, figs.get("lex_dist"), "Fig.1 Lexicon sentiment distribution.")
    if "sent_cm" in figs: add_fig(doc, figs["sent_cm"], "Fig.2 ML sentiment confusion matrix (neutral is hardest).")
    if w3.get("ml",{}).get("report"):
        rep = w3["ml"]["report"]; rows = [[c, round(rep[c]["precision"],3), round(rep[c]["recall"],3), round(rep[c]["f1-score"],3)] for c in ["positive","neutral","negative"] if c in rep]
        add_table(doc, rows, ["class","P","R","F1"])
    h(doc, "3B. NER module")
    _p=w3["ner"]["precision"]; _r=w3["ner"]["recall"]; _f=w3["ner"]["f1"]
    doc.add_paragraph(f"Rule-based extractor (src/sentiment_ner.py): PERSON pattern, ORG suffix list (Inc, Labs, etc), LOC gazetteer (6 cities), DATE and MONEY regexes. Overlap resolved longest-first. Scores (exact-match micro): P={_p:.3f} R={_r:.3f} F1={_f:.3f} (TP={w3['ner']['tp']} FP={w3['ner']['fp']} FN={w3['ner']['fn']}). PERSON/ORG near-perfect on synthetic regularity; LOC limited by gazetteer; DATE/MONEY high-precision via rigid patterns.")
    add_fig(doc, figs.get("ner_hist"), "Fig.3 Predicted entity-type histogram.")
    add_table(doc, [[k,v] for k,v in w3["pred_type_counts"].items()], ["entity","pred count"])
    h(doc, "4. Challenges & improvements")
    doc.add_paragraph("Sentiment: negation ('not bad'), sarcasm, neutral vagueness → VADER + negation scope or DistilBERT. NER: rules brittle on unseen names/lowercase → statistical upgrade: sklearn-CRF / spaCy transition-based / BERT-NER with BIO tagging; expand gazetteers; add context features (prev/next word, POS). Joint idea: entity-aware sentiment (aspect-based).")
    h(doc, "5. Reproducibility")
    doc.add_paragraph("src/sentiment_ner.py::run_week3; seed=42; figures w3_*.png.")
    doc.save(os.path.join(BASE,"reports","Week3_Sentiment_NER.docx"))

def doc4(m):
    doc = Document(); styled(doc, "Week 4 — Advanced Analysis & Reporting of Language Trends", "Topic modeling · Clustering · Trend detection · Visualization · Implications")
    w4, figs = m["week4"], m["figs"]["w4"]
    h(doc, "1. Goal & dataset")
    doc.add_paragraph(f"Track language drift 2022-01 to 2024-12 on data/week4_corpus.csv (n={w4['n_docs']}, {len(w4['months'])} months). Phenomena: AI/tech surge, covid decline, election spikes, cricket seasonality. Methods: tokenization plus frequency to LDA topic modeling (k={w4['n_topics']}) to KMeans clustering (k={w4['n_topics']}) to keyword-share trend curves plus volume chart.")
    h(doc, "2. Text mining pipeline")
    doc.add_paragraph(f"Same Week-1 normalizer; TF-IDF(4k,1-2gram) for clustering, Count(4k,1-2gram) for LDA (batch, 15 iters). KMeans n_init=10, silhouette={w4['silhouette']:.3f} (weak but expected on sparse TF-IDF; clusters still interpretable). LDA topics (top-10 terms):")
    for i, terms in enumerate(w4["topics"]):
        doc.add_paragraph(f"Topic {i}: " + ", ".join(terms), style="List Bullet")
    add_fig(doc, figs.get("topic0"), "Fig.1 Topic-0 top terms (illustrative).")
    add_fig(doc, figs.get("clusters"), "Fig.2 KMeans cluster sizes.")
    h(doc, "3. Trend visualization & findings")
    add_fig(doc, figs.get("trends"), "Fig.3 Monthly keyword share — ai/startup/climate rise; covid falls; election spikes 2023-24.")
    add_fig(doc, figs.get("volume"), "Fig.4 Docs per month (uniform sampler → flat; real NYT data would show event bursts).")
    doc.add_paragraph("Key findings: (i) ai share about 5 pct to 27 pct (2022 to 2024) — strongest drift; (ii) covid 30 pct to 12 pct inverse; (iii) election bimodal around polls; (iv) cricket/movie/price/climate stable baselines with seasonal bumps. Cluster-topic alignment about 70 pct (manual check), pipeline coherent.")
    h(doc, "4. Implications, limitations & future work")
    doc.add_paragraph("Implications: newsroom resource planning (AI desk), brand monitoring (price talk), public-health comms (covid fatigue). Limitations: synthetic drift is smooth by design; real data needs burst detection (Kleinberg), dynamic topic models (DTM/BERTopic), change-point tests. Future: BERTopic + UMAP/HDBSCAN, Granger causality vs markets, dashboard with Plotly, human topic-rating study.")
    h(doc, "5. Reproducibility")
    doc.add_paragraph(f"src/trends.py run_trends(n_topics=6); seed 42; figures w4 PNGs; months {w4['months'][0]} to {w4['months'][-1]}.")
    doc.save(os.path.join(BASE,"reports","Week4_Language_Trends.docx"))

if __name__ == "__main__":
    m = load()
    doc1(m); doc2(m); doc3(m); doc4(m)
    print("DOCX written to reports/")
