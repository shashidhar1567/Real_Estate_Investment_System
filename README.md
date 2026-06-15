# 🏠 Real Estate Investment Potential Recommender

A Streamlit web app that predicts whether a property has **High**, **Medium**, or **Low** investment potential using two ML models tuned with GridSearchCV.

## 📁 Project Structure
```
real_estate_investment_app/
├── app.py               ← Main Streamlit application
├── model_rf.pkl         ← Tuned Random Forest model
├── model_lr.pkl         ← Tuned Logistic Regression model
├── scaler.pkl           ← StandardScaler fitted on training data
├── hpt_summary.json     ← HPT results & classification reports
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud (FREE)
1. Push this folder to a **GitHub repository**
2. Go to → https://share.streamlit.io
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Click **Deploy** → get a public URL instantly ✅

## ☁️ Deploy on Hugging Face Spaces (FREE)
1. Create a new Space at https://huggingface.co/spaces
2. Choose SDK = **Streamlit**
3. Upload all 6 files → Space auto-builds ✅

## 🧠 Models
| Model | Algorithm | Test Accuracy |
|-------|-----------|---------------|
| model_rf.pkl | Random Forest (HPT) | **95.61%** 🏆 |
| model_lr.pkl | Logistic Regression (HPT) | 75.49% |

## 🔧 Hyperparameter Tuning
- Method: **GridSearchCV** with **5-Fold Cross Validation**
- RF best params: `max_depth=15, n_estimators=100, min_samples_split=2`
- LR best params: `C=10, solver=lbfgs, max_iter=500`

## 📊 App Tabs
- **🏠 Prediction** — Enter property details, choose a model, get prediction + probabilities
- **📊 Model Comparison** — Accuracy, precision, recall, F1 comparison charts
- **🔧 Hyperparameter Tuning** — Grid details, best params, before vs after accuracy
