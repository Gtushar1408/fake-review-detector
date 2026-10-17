import os, requests, streamlit as st
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.title("🕵️ Fake Review Detector")
img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
text = st.text_area("Review text")
rating = st.select_slider("Star rating", [1, 2, 3, 4, 5], value=5)
reviewer = st.text_input("Your name (optional)")
bid = st.text_input("Institute Name (try: demo)")
place = st.text_input("Location name, e.g. city (optional)")
thr = st.slider("Majority-outlier threshold", 0.5, 1.0, 0.8)
rule = st.checkbox("Enable severe-abuse rule", True)

if st.button("Analyze Review"):
    if not img or not text:
        st.error("Please upload an image and write a review.")
    else:
        data = {"review_text": text, "rating": rating, "majority_threshold": thr, "severe_rule_enabled": rule}
        if bid: data["business_id"] = bid
        if reviewer: data["reviewer_name"] = reviewer
        if place: data["location_name"] = place
        try:
            r = requests.post(f"{API_URL}/predict",
                              files={"image": (img.name, img.getvalue(), img.type)}, data=data, timeout=90)
        except Exception as e:
            st.error(f"Backend not reachable: {e}"); st.stop()
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            j = r.json()
            st.subheader(f"Prediction: {j['prediction']}  (risk {j['risk_score']}/100)")
            st.progress(j["risk_score"] / 100)
            st.markdown("**Input facts / metadata**")
            st.write("📍", j["image_gps_status"])
            st.write("Rating matches sentiment:", j["rating_sentiment_match"])
            st.markdown("**Model output (heuristic, demo data)**")
            st.write(j["heuristic_ml_score"])
            st.markdown("**Heuristic flags**")
            for s in j["detected_signals"] or ["None"]: st.write("•", s)
            st.markdown("**Majority analysis**"); st.json(j["majority_analysis"])
            st.info(j["explanation"])