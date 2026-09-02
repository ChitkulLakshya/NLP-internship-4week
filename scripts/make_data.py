"""Deterministic synthetic data generator (seed=42). No network calls."""
import os, random, datetime
import pandas as pd
BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")
random.seed(42)

TECH = ["phone battery life is {adj}", "the new AI startup raised funding for {noun}", "laptop performance with {noun} is {adj}",
        "software update fixed {noun} but battery still {adj}", "I love the camera quality, pictures look {adj}"]
SPORTS = ["cricket match was {adj}, {team} won by {n} runs", "{team} played {adj} football last night",
          "the stadium crowd went wild, game was {adj}", "coach praised {team} after {adj} victory"]
POLITICS = ["election debate on {noun} was {adj}", "parliament passed {noun} bill amid {adj} protests",
            "minister promised {adj} reforms for {noun}", "voters reacted {adj} to new {noun} policy"]
ENT = ["the movie was {adj}, songs were {adj}", "new web series on {noun} is {adj}", "concert last night was {adj}, crowd {adj}",
       "actor performance in {noun} felt {adj}"]
ADJ_POS = ["amazing","great","fantastic","excellent","wonderful","awesome","brilliant"]
ADJ_NEG = ["terrible","awful","boring","disappointing","poor","horrible","bad"]
ADJ_NEU = ["okay","average","decent","fine","mixed"]
NOUNS = ["battery","camera","script","policy","season","album","match","phone","laptop","budget"]
TEAMS = ["India","MI","CSK","RCB","Hyderabad FC"]

REVIEW_TPL = [
    "I bought {prod} last week and it is {adj}. {extra}",
    "Honestly the {prod} feels {adj} — {extra}",
    "{extra} Overall {prod} is {adj}.",
    "My experience with {prod}: {adj}! {extra}",
]
PRODS = ["phone","laptop","headphones","camera","mixer","shoes","watch","book","course","app"]
EXTRAS = ["Delivery was quick.", "Check this https://example.com/deal for details.",
          "Contact @support was helpful #happy 😊", "<p>Highly recommended!</p>",
          "SOOOO good!!! Loved it!!!", "meh, visit www.shop.example?", ""]

def maybe_noise(s, p=0.22):
    if random.random() < p:
        s += " " + random.choice(EXTRAS)
    if random.random() < 0.06:
        s = s.replace("o", "ooo", 1) if "o" in s else s
    return s

def sentiment_of(adj):
    if adj in ADJ_POS: return "positive"
    if adj in ADJ_NEG: return "negative"
    return "neutral"

def gen_week1(n=1500):
    rows = []
    for i in range(n):
        src = random.choices(["twitter","amazon","news"], weights=[0.45,0.35,0.2])[0]
        if src == "twitter":
            adj = random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU)
            t = f"{random.choice(['Loving','Hating','Just tried'])} {random.choice(PRODS)} — {adj}! {'@user' if random.random()<0.2 else ''} {'#review' if random.random()<0.25 else ''} {'https://t.co/xyz' if random.random()<0.2 else ''} {'😊' if adj in ADJ_POS else ('😡' if adj in ADJ_NEG else '')}"
        elif src == "amazon":
            prod = random.choice(PRODS); adj = random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU)
            t = random.choice(REVIEW_TPL).format(prod=prod, adj=adj, extra=random.choice(EXTRAS))
            t = maybe_noise(t)
        else:
            cat = random.choice([TECH,SPORTS,POLITICS,ENT]); tpl = random.choice(cat)
            adj = random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU)
            t = tpl.format(adj=adj, noun=random.choice(NOUNS), team=random.choice(TEAMS), n=random.randint(1,80), prod=random.choice(PRODS))
            t = maybe_noise(t, p=0.12)
        rating = {"positive": random.choices([4,5],[0.35,0.65])[0], "negative": random.choices([1,2],[0.6,0.4])[0], "neutral": 3}[sentiment_of(adj)]
        if random.random() < 0.03: t = ""  # missing
        if random.random() < 0.02: t = None
        d = datetime.date(2023,1,1) + datetime.timedelta(days=random.randint(0,700))
        rows.append({"id": i+1, "date": str(d), "source": src, "text": t, "rating": rating})
    pd.DataFrame(rows).to_csv(os.path.join(DATA,"week1_reviews.csv"), index=False)

def gen_week2(n=2000):
    rows = []
    cats = [("tech",TECH),("sports",SPORTS),("politics",POLITICS),("entertainment",ENT)]
    for i in range(n):
        label, bank = random.choice(cats)
        tpl = random.choice(bank)
        adj = random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU+["innovative","record","historic"])
        t = tpl.format(adj=adj, noun=random.choice(NOUNS), team=random.choice(TEAMS), n=random.randint(1,100))
        if random.random()<0.15: t += " " + random.choice(EXTRAS)
        rows.append({"id": i+1, "text": t, "label": label})
    pd.DataFrame(rows).to_csv(os.path.join(DATA,"week2_news.csv"), index=False)

PEOPLE = ["Ananya Sharma","Rahul Verma","Priya Nair","Arjun Reddy","Sneha Kulkarni","Vikram Rao"]
ORGS = ["Growon Labs","TechNova Inc","Zenith Systems","FreshKart Foods","Skyline Airlines","PixelWorks Studio"]
PLACES = ["Hyderabad","Mumbai","Delhi","Bangalore","London","New York"]
def gen_week3(n=1200):
    rows=[]
    for i in range(n):
        prod = random.choice(PRODS); adj = random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU)
        person = random.choice(PEOPLE); org = random.choice(ORGS); loc = random.choice(PLACES)
        money = f"${random.randint(10,2000)}" if random.random()<0.4 else f"{random.randint(500,50000)} rupees"
        date = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/2024" if random.random()<0.35 else ""
        sent = sentiment_of(adj)
        t = f"{person} reviewed {prod} from {org} in {loc}. It felt {adj} and cost {money}. {date} {'Highly recommend!' if sent=='positive' else ('Total waste.' if sent=='negative' else 'It is okay.')}"
        if random.random()<0.15: t += " https://example.com/review @seller #genuine 😊"
        ents=[{"text":person,"label":"PERSON"},{"text":org,"label":"ORG"},{"text":loc,"label":"LOC"},{"text":money,"label":"MONEY"}]
        if date: ents.append({"text":date,"label":"DATE"})
        rows.append({"id":i+1,"text":t,"sentiment":sent,"entities":str(ents)})
    pd.DataFrame(rows).to_csv(os.path.join(DATA,"week3_reviews.csv"), index=False)

def gen_week4(n=1800):
    rows=[]; start=datetime.date(2022,1,1)
    for i in range(n):
        m = random.randint(0,35); d = start + datetime.timedelta(days=30*m+random.randint(0,27))
        year_frac = m/35.0
        # drift: ai/startup/climate up over time; covid down; election spikes mid-2023/2024; cricket seasonal
        r = random.random()
        if r < 0.05+0.22*year_frac:
            t = f"AI startup raises funding for {random.choice(NOUNS)}, models look {random.choice(ADJ_POS+['innovative'])}"
        elif r < 0.30-0.18*year_frac:
            t = f"covid cases and {random.choice(NOUNS)} update, situation {random.choice(ADJ_NEU+ADJ_NEG)}"
        elif r < 0.45:
            t = f"election {random.choice(['debate','rally','poll'])} on {random.choice(NOUNS)} draws {random.choice(ADJ_NEU+ADJ_POS)} response"
        elif r < 0.60:
            t = f"cricket: {random.choice(TEAMS)} wins thriller, fans call it {random.choice(ADJ_POS)}"
        elif r < 0.72:
            t = f"movie review: {random.choice(NOUNS)} storyline feels {random.choice(ADJ_POS+ADJ_NEG+ADJ_NEU)}"
        elif r < 0.85:
            t = f"price of {random.choice(NOUNS)} hits {random.choice(['record high','new low'])} amid {random.choice(['inflation','demand'])}"
        else:
            t = f"climate summit pushes {random.choice(NOUNS)} action, activists {random.choice(['hopeful','critical'])}"
        if random.random()<0.1: t += " https://news.example/link"
        rows.append({"id":i+1,"date":str(d),"text":t})
    pd.DataFrame(rows).sort_values("date").to_csv(os.path.join(DATA,"week4_corpus.csv"), index=False)

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    gen_week1(); gen_week2(); gen_week3(); gen_week4()
    print("data written to", DATA)
