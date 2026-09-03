"""Run all 4 pipelines; save figures + metrics.json."""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.utils import seed_everything, dump_json
seed_everything(42)
from src.eda import run_eda_week1
from src.classification import run_classification
from src.sentiment_ner import run_week3
from src.trends import run_trends

FIG = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(BASE, "data")

if __name__ == "__main__":
    print("Week1 EDA...")
    s1, f1, _ = run_eda_week1(os.path.join(DATA,"week1_reviews.csv"), FIG)
    print("Week2 classification...")
    s2, f2 = run_classification(os.path.join(DATA,"week2_news.csv"), FIG)
    print("Week3 sentiment+NER...")
    s3, f3, _ = run_week3(os.path.join(DATA,"week3_reviews.csv"), FIG)
    print("Week4 trends...")
    s4, f4, _ = run_trends(os.path.join(DATA,"week4_corpus.csv"), FIG)
    metrics = {"week1": s1, "week2": s2, "week3": s3, "week4": s4,
               "figs": {"w1": f1, "w2": f2, "w3": f3, "w4": f4}}
    # stringify non-serializable
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k,v in o.items()}
        if isinstance(o, (list,tuple)): return [clean(v) for v in o]
        if isinstance(o, float):
            if o != o: return 0.0
            return o
        try:
            json.dumps(o); return o
        except Exception:
            return str(o)
    dump_json(clean(metrics), os.path.join(BASE,"reports","metrics.json"))
    print("DONE. metrics + figures in reports/")
