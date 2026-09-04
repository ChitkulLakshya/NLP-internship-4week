import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

def test_classification_runs():
    from src.classification import run_classification
    import tempfile
    from scripts.make_data import gen_week2  # ensures generator importable
    s, figs = run_classification(os.path.join(BASE,"data","week2_news.csv"), tempfile.mkdtemp())
    assert s["models"]["LinearSVC"]["test_f1_macro"] > 0.80
    assert set(s["labels"]) == {"entertainment","politics","sports","tech"}

def test_eda_runs():
    from src.eda import run_eda_week1
    import tempfile
    s, figs, _ = run_eda_week1(os.path.join(BASE,"data","week1_reviews.csv"), tempfile.mkdtemp())
    assert s["n_rows"] >= 1000 and len(s["top_terms"]) > 5
