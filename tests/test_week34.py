import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.sentiment_ner import lex_predict, extract_entities, score_ner

def test_lex_positive():
    assert lex_predict("I love this, amazing and fantastic!") == "positive"

def test_lex_negative():
    assert lex_predict("Terrible awful waste, hate it") == "negative"

def test_ner_extracts_person_org():
    ents = extract_entities("Ananya Sharma from Growon Labs in Hyderabad won $500 on 12/05/2024")
    labels = {e["label"] for e in ents}
    assert {"PERSON","ORG","LOC","MONEY","DATE"} & labels

def test_ner_scoring():
    p = [[{"text":"A B","label":"PERSON"}]]
    g = [[{"text":"A B","label":"PERSON"}]]
    assert score_ner(p,g)["f1"] == 1.0

def test_week3_runs():
    import tempfile
    from src.sentiment_ner import run_week3
    s, figs, _ = run_week3(os.path.join(BASE,"data","week3_reviews.csv"), tempfile.mkdtemp())
    assert s["ner"]["f1"] > 0.5 and s["ml"]["acc"] > 0.80

def test_week4_runs():
    import tempfile
    from src.trends import run_trends
    s, figs, _ = run_trends(os.path.join(BASE,"data","week4_corpus.csv"), tempfile.mkdtemp(), n_topics=4)
    assert len(s["topics"]) == 4 and "ai" in s["trend"]
