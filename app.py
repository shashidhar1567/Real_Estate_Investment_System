import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Real Estate Investment Recommender",
    page_icon="🏠",
    layout="wide"
)

# ─── Load artefacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf  = joblib.load("model_rf.pkl")
    lr  = joblib.load("model_lr.pkl")
    sc  = joblib.load("scaler.pkl")
    with open("hpt_summary.json") as f:
        hpt = json.load(f)
    return rf, lr, sc, hpt

model_rf, model_lr, scaler, hpt = load_artifacts()

# ─── Encoding maps ────────────────────────────────────────────────────────────
LOCALITY_MAP    = {"Bridgeport":0,"Fairfield":1,"Greenwich":2,"Norwalk":3,
                   "Stamford":4,"Unknown":5,"Waterbury":6,"West Hartford":7}
PROPERTY_MAP    = {"Single Family":2,"Two Family":4,"Three Family":3,"Four Family":1,"Other":0}
RESIDENTIAL_MAP = {"Detached House":0,"Duplex":1,"Triplex":3,"Fourplex":2}
FACE_MAP        = {"North":1,"South":2,"East":0,"West":3}
LABEL_COLORS    = {"High":"#22c55e","Medium":"#f59e0b","Low":"#ef4444"}
MODEL_OPTIONS   = {"Random Forest (Best)":"rf","Logistic Regression":"lr"}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Predict a Property")
    chosen = st.radio("Select Model", list(MODEL_OPTIONS.keys()))

    st.markdown("---")
    locality      = st.selectbox("Locality",          list(LOCALITY_MAP.keys()))
    property_type = st.selectbox("Property Type",     list(PROPERTY_MAP.keys()))
    residential   = st.selectbox("Residential Type",  list(RESIDENTIAL_MAP.keys()))
    face          = st.selectbox("Facing Direction",   list(FACE_MAP.keys()))
    year          = st.number_input("Year", 2009, 2030, 2022)
    est_val       = st.number_input("Estimated Value ($)", 0.0, 30_000_000.0, 250_000.0, 1000.0)
    sale_price    = st.number_input("Sale / Asking Price ($)", 0.0, 30_000_000.0, 350_000.0, 1000.0)
    num_rooms     = st.number_input("Rooms", 1, 20, 3)
    num_bath      = st.number_input("Bathrooms", 1, 20, 2)
    carpet        = st.number_input("Carpet Area (sq ft)", 100.0, 5000.0, 1000.0, 10.0)
    tax_rate      = st.number_input("Property Tax Rate", 1.0, 2.0, 1.03, 0.001, format="%.4f")

    predict_btn   = st.button("🔍 Predict", use_container_width=True, type="primary")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Model Comparison", "🔧 Hyperparameter Tuning"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🏠 Real Estate Investment Potential Recommender")
    st.caption("Predict whether a property has **High**, **Medium**, or **Low** investment potential.")

    if predict_btn:
        price_ratio = sale_price / est_val if est_val > 0 else 0

        feat = pd.DataFrame([{
            "Year": year, "Estimated Value": est_val,
            "num_rooms": num_rooms, "num_bathrooms": num_bath,
            "carpet_area": carpet, "property_tax_rate": tax_rate,
            "price_ratio": price_ratio,
            "Locality_enc":    LOCALITY_MAP[locality],
            "Property_enc":    PROPERTY_MAP[property_type],
            "Residential_enc": RESIDENTIAL_MAP[residential],
            "Face_enc":        FACE_MAP[face],
        }])

        feat_sc = scaler.transform(feat)
        mdl     = model_rf if MODEL_OPTIONS[chosen] == "rf" else model_lr
        pred    = mdl.predict(feat_sc)[0]
        proba   = mdl.predict_proba(feat_sc)[0]
        classes = mdl.classes_

        c1, c2 = st.columns([1, 1])
        with c1:
            color = LABEL_COLORS.get(pred, "#6b7280")
            st.markdown(f"""
            <div style="padding:1.4rem;border-radius:.75rem;background:{color}18;border:2px solid {color};margin-bottom:1rem">
                <h2 style="color:{color};margin:0">Investment Potential: {pred}</h2>
                <p style="margin:.4rem 0 0">Model: <b>{chosen}</b> &nbsp;|&nbsp; Price Ratio: <b>{price_ratio:.3f}</b></p>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Class Probabilities")
            prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Class"), color="#6366f1")

            if pred == "High":
                st.success("✅ Strong investment potential — premium price tier.")
            elif pred == "Medium":
                st.warning("⚠️ Moderate potential — consider negotiating or checking growth trends.")
            else:
                st.info("ℹ️ Lower tier — better suited for budget buyers or rental income.")

        with c2:
            st.markdown("#### Input Summary")
            st.json({
                "Locality": locality, "Property Type": property_type,
                "Residential": residential, "Facing": face,
                "Year": year, "Estimated Value ($)": est_val,
                "Sale Price ($)": sale_price, "Price Ratio": round(price_ratio, 4),
                "Rooms": int(num_rooms), "Bathrooms": int(num_bath),
                "Carpet Area (sqft)": carpet, "Tax Rate": tax_rate,
            })
    else:
        st.info("👈 Fill in property details in the sidebar and click **Predict**.")
        st.markdown("""
        ### How it works
        | Step | Detail |
        |------|--------|
        | Input | 11 property features (location, size, value, facing, …) |
        | Target | `High` (≥ $400K) · `Medium` ($150K–$400K) · `Low` (< $150K) sale price tier |
        | Models | Logistic Regression · Random Forest (both tuned via GridSearchCV) |
        | Winner | **Random Forest** — 95.6% test accuracy |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Model Comparison")

    lr_acc = hpt["lr"]["acc"]
    rf_acc = hpt["rf"]["acc"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Logistic Regression Accuracy", f"{lr_acc*100:.2f}%")
    m2.metric("Random Forest Accuracy",       f"{rf_acc*100:.2f}%", delta=f"+{(rf_acc-lr_acc)*100:.2f}%")
    m3.metric("Winner", "Random Forest 🏆")

    st.markdown("---")
    st.subheader("Accuracy Comparison")
    acc_df = pd.DataFrame({
        "Model":    ["Logistic Regression", "Random Forest"],
        "Accuracy": [lr_acc, rf_acc],
    }).set_index("Model")
    st.bar_chart(acc_df, color="#6366f1")

    st.markdown("---")
    c1, c2 = st.columns(2)

    for col, key, label in [(c1, "lr", "Logistic Regression"), (c2, "rf", "Random Forest")]:
        with col:
            st.subheader(f"📋 {label}")
            report = hpt[key]["report"]
            rows = []
            for cls in ["High", "Medium", "Low"]:
                r = report[cls]
                rows.append({
                    "Class":     cls,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall":    f"{r['recall']:.3f}",
                    "F1-Score":  f"{r['f1-score']:.3f}",
                    "Support":   int(r["support"]),
                })
            st.table(pd.DataFrame(rows).set_index("Class"))

            wa = report["weighted avg"]
            st.markdown(f"""
            **Weighted Avg** &nbsp;→&nbsp;
            Precision `{wa['precision']:.3f}` · Recall `{wa['recall']:.3f}` · F1 `{wa['f1-score']:.3f}`
            """)

    st.markdown("---")
    st.subheader("Per-class F1 Score Comparison")
    f1_data = {}
    for cls in ["High", "Medium", "Low"]:
        f1_data[cls] = {
            "Logistic Regression": hpt["lr"]["report"][cls]["f1-score"],
            "Random Forest":       hpt["rf"]["report"][cls]["f1-score"],
        }
    f1_df = pd.DataFrame(f1_data).T
    st.bar_chart(f1_df, color=["#6366f1", "#22c55e"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔧 Hyperparameter Tuning (GridSearchCV, 5-Fold CV)")
    st.markdown("""
    Both models were tuned using **GridSearchCV** with **5-fold cross-validation**
    on the training set. The best parameters per model are shown below.
    """)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Logistic Regression")
        st.markdown("**Grid searched:**")
        st.json({"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"], "max_iter": [500, 1000]})
        st.markdown("**Best parameters found:**")
        st.success(str(hpt["lr"]["params"]))
        st.metric("Post-tuning Test Accuracy", f"{hpt['lr']['acc']*100:.2f}%")
        st.markdown("""
        **Key insight:** Logistic Regression struggles with non-linear boundaries
        between investment tiers. Higher `C = 10` reduces regularisation,
        fitting the data more aggressively, but the model still caps at ~75%.
        """)

    with c2:
        st.subheader("Random Forest")
        st.markdown("**Grid searched:**")
        st.json({"n_estimators": [50, 100, 200], "max_depth": [6, 10, 15], "min_samples_split": [2, 5]})
        st.markdown("**Best parameters found:**")
        st.success(str(hpt["rf"]["params"]))
        st.metric("Post-tuning Test Accuracy", f"{hpt['rf']['acc']*100:.2f}%")
        st.markdown("""
        **Key insight:** Deeper trees (`max_depth = 15`) with fine splits
        (`min_samples_split = 2`) allow the forest to capture complex interactions
        between price ratio, locality, and carpet area — boosting accuracy to **95.6%**.
        """)

    st.markdown("---")
    st.subheader("Accuracy Before vs After Tuning")
    tuning_df = pd.DataFrame({
        "Model":        ["Logistic Regression", "Random Forest"],
        "Before HPT":   [0.7549, 0.9443],
        "After HPT":    [hpt["lr"]["acc"], hpt["rf"]["acc"]],
    }).set_index("Model")
    st.bar_chart(tuning_df, color=["#94a3b8", "#6366f1"])

    st.markdown("---")
    st.subheader("📝 Conclusion")
    st.info("""
    **Random Forest with HPT** is the clear winner:

    - ✅ **95.6% accuracy** vs 75.5% for Logistic Regression  
    - ✅ High F1-score across all 3 classes (High / Medium / Low)  
    - ✅ Best hyperparameters: `max_depth=15`, `min_samples_split=2`, `n_estimators=100`  
    - ℹ️ Logistic Regression is better for interpretability but unsuitable for the
      non-linear patterns in this dataset.
    """)
