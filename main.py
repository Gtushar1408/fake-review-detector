import io, re, math
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Fake Review Detector (demo)")

# ---------- CONFIG ----------
SEVERE_WORDS = {"motherfucker", "fuck", "fucking", "fucker", "cunt", "bastard", "asshole", "shit"}  # edit freely
MILD_WORDS = {"damn", "hell", "crap", "stupid", "idiot"}
MILD_LIMIT = 3                 # mild words needed to count as abusive
MIN_HISTORY = 5                # min reviews for majority analysis
PROMO = ["best ever", "must buy", "buy now", "100% recommend", "click here", "discount", "free gift", "visit our"]
POS = {"good","great","excellent","amazing","love","loved","tasty","friendly","best","awesome","perfect","nice","clean","fresh"}
NEG = {"bad","terrible","awful","worst","hate","hated","rude","dirty","slow","cold","poor","disgusting","horrible","stale"}

def norm(t):
    t = t.lower().translate(str.maketrans("@$0134!*", "asoieaii"))  # simple leetspeak
    return re.sub(r"[^a-z\s]", "", t)

def sentiment(t):
    w = norm(t).split()
    p, n = sum(x in POS for x in w), sum(x in NEG for x in w)
    return (p - n) / max(1, p + n)

# ---------- DEMO TRAINING DATA (SYNTHETIC) ----------
TRAIN = [
 ("Great food and friendly staff, loved the pasta", 5, 0), ("Service was slow and the soup was cold", 2, 0),
 ("Nice clean place, good coffee", 4, 0), ("Terrible experience, rude waiter and dirty tables", 1, 0),
 ("Perfect evening, fresh and tasty dishes", 5, 0), ("Average meal, nothing special but okay", 3, 0),
 ("BEST EVER!!! MUST BUY!!! 100% recommend buy now", 5, 1), ("Amazing amazing amazing amazing amazing place amazing", 5, 1),
 ("Worst place ever!!! avoid!!! total scam!!!", 1, 1), ("Click here for free gift best ever visit our page", 5, 1),
 ("Great great great great great great", 5, 1), ("Awful awful awful horrible horrible horrible!!!", 1, 1),
 ("Best product best price best best best", 5, 1), ("Good food", 1, 1), ("Terrible food and I loved it, amazing", 1, 1),
]
vec = TfidfVectorizer(ngram_range=(1, 2))
def num(text, rating): return [rating / 5, sentiment(text), min(len(text), 500) / 500]
Xt = vec.fit_transform([t for t, _, _ in TRAIN])
X = hstack([Xt, csr_matrix([num(t, r) for t, r, _ in TRAIN])]).tocsr()
clf = LogisticRegression(max_iter=1000).fit(X, [y for _, _, y in TRAIN])

# ---------- DEMO BUSINESS HISTORY (business_id="demo": 90 positive, 10 negative) ----------
adj = ["Great","Lovely","Excellent","Tasty","Fresh","Friendly","Awesome","Perfect","Nice"]
noun = ["food","service","pizza","coffee","staff","desserts","ambience","burgers","pasta","breakfast"]
HISTORY = {"demo": [(f"{a} {n} here, enjoyed visit", 5) for a in adj for n in noun]
                   + [(f"Terrible {n}, very bad and rude", 1) for n in noun]}

# ---------- HELPERS ----------
def get_gps(img):
    try:
        g = img.getexif().get_ifd(0x8825)
        if not g or 2 not in g or 4 not in g: return None
        d = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat, lon = d(g[2]), d(g[4])
        if g.get(1) == "S": lat = -lat
        if g.get(3) == "W": lon = -lon
        return lat, lon
    except Exception:
        return None

def km(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c-a)*p/2)**2 + math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2
    return 12742 * math.asin(math.sqrt(x))

# ---------- ENDPOINT ----------
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    review_text: str = Form(...),
    rating: float = Form(...),
    business_id: Optional[str] = Form(None),
    business_latitude: Optional[float] = Form(None),
    business_longitude: Optional[float] = Form(None),
    majority_threshold: float = Form(0.8),          # majority share needed to call it "majority"
    severe_rule_enabled: bool = Form(True),
):
    if not (1 <= rating <= 5): raise HTTPException(400, "rating must be 1-5")
    if len(review_text.strip()) < 3: raise HTTPException(400, "review_text too short")
    if image.content_type not in ("image/jpeg", "image/png"): raise HTTPException(400, "Only JPEG/PNG allowed")
    data = await image.read()
    if len(data) > 5 * 1024 * 1024: raise HTTPException(400, "Image over 5MB")
    try: img = Image.open(io.BytesIO(data))
    except Exception: raise HTTPException(400, "Invalid image file")

    signals, score = [], 0
    words = norm(review_text).split()
    sent = sentiment(review_text)

    # ML part
    xv = hstack([vec.transform([review_text]), csr_matrix([num(review_text, rating)])]).tocsr()
    prob = float(clf.predict_proba(xv)[0][1])
    score += prob * 40

    # Rule A: severe abusive language
    severe = [w for w in words if w in SEVERE_WORDS]
    mild = [w for w in words if w in MILD_WORDS]
    rule_a = severe_rule_enabled and (len(severe) >= 1 or len(mild) >= MILD_LIMIT)
    if rule_a: signals.append("Flagged by the custom severe-abusive-language rule.")

    # Rating vs sentiment
    mismatch = (sent > 0.3 and rating <= 2) or (sent < -0.3 and rating >= 4)
    if mismatch: signals.append("Rating contradicts review sentiment"); score += 25
    if rating in (1, 5): signals.append("Extreme rating (weak signal)"); score += 5

    # Text patterns
    letters = [c for c in review_text if c.isalpha()]
    if len(letters) > 10 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        signals.append("Excessive capitalization"); score += 10
    if review_text.count("!") >= 3: signals.append("Excessive punctuation"); score += 10
    if any(p in review_text.lower() for p in PROMO): signals.append("Promotional wording"); score += 15
    if words and Counter(words).most_common(1)[0][1] >= 4: signals.append("Repeated words"); score += 10
    if len(words) < 3: signals.append("Very short review"); score += 5

    # GPS
    gps = get_gps(img)
    if gps is None:
        gps_status = "GPS metadata not available (missing evidence, not proof of fraud)"
    else:
        gps_status = f"GPS found: {gps[0]:.5f}, {gps[1]:.5f}"
        if business_latitude is not None and business_longitude is not None:
            dist = km(gps[0], gps[1], business_latitude, business_longitude)
            gps_status += f" | {dist:.1f} km from business"
            if dist > 1: signals.append(f"Photo taken {dist:.1f} km from business"); score += 15

    # Duplicate + majority analysis (needs history)
    hist = HISTORY.get(business_id or "", [])
    majority = {"available": False, "note": "No historical reviews - majority analysis cannot be done reliably."}
    dup = any(SequenceMatcher(None, review_text.lower(), t.lower()).ratio() > 0.9 for t, _ in hist)
    if dup: signals.append("Duplicate/near-duplicate of existing review"); score += 25
    outlier = False
    if len(hist) >= MIN_HISTORY:
        pos = sum(r >= 4 for _, r in hist) / len(hist)
        neg = sum(r <= 2 for _, r in hist) / len(hist)
        mine = "pos" if rating >= 4 else "neg" if rating <= 2 else "neutral"
        outlier = (mine == "neg" and pos >= majority_threshold) or (mine == "pos" and neg >= majority_threshold)
        majority = {"available": True, "reviews": len(hist), "positive_share": round(pos, 2),
                    "negative_share": round(neg, 2), "threshold": majority_threshold, "outlier": outlier}
        if outlier: signals.append("Majority-outlier (opposes majority opinion)"); score += 15
    elif business_id:
        majority["note"] = "Not enough history for this business_id."

    score = min(100, round(score))
    other_evidence = rule_a or mismatch or dup or len(signals) - outlier >= 3

    if rule_a: pred, score = "FAKE", max(score, 90)
    elif outlier and other_evidence: pred = "FAKE" if score >= 60 else "NEEDS_REVIEW"
    elif outlier: pred = "NEEDS_REVIEW"
    elif score >= 60: pred = "FAKE"
    elif score >= 35: pred = "NEEDS_REVIEW"
    else: pred = "GENUINE"

    return {
        "prediction": pred,
        "heuristic_ml_score": round(prob, 3),  # NOT calibrated, trained on tiny synthetic data
        "risk_score": score,
        "detected_signals": signals,
        "image_gps_status": gps_status,
        "rating_sentiment_match": not mismatch,
        "majority_analysis": majority,
        "explanation": f"{pred}: risk {score}/100. " + ("; ".join(signals) if signals else "No suspicious signals found."),
    }