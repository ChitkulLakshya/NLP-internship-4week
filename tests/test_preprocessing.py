import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.preprocessing import clean_text, tokenize, preprocess, detect_noise

def test_clean_strips_url_and_mention():
    assert "http" not in clean_text("see https://ex.com @user hi")
    assert "user" not in clean_text("hi @user")

def test_hashtag_keeps_word():
    assert "happy" in clean_text("feeling #happy today")

def test_emoji_preserved_as_token():
    assert "<emoji>" in clean_text("nice 😊")

def test_tokenize_basic():
    toks = tokenize("Hello, world! It's great.")
    assert "hello" in toks and "world" in toks

def test_preprocess_pipeline():
    out = preprocess("SOOOO good!!! https://ex.com #happy")
    assert isinstance(out["normalized"], list) and len(out["normalized"]) > 0

def test_noise_detector():
    f = detect_noise("hi https://ex.com @u #t")
    assert f["has_url"] and f["has_mention"] and f["has_hashtag"]
