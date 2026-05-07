# 🧠 Dyslexia Detector — Streamlit App

A complete multi-page Streamlit application for dyslexia screening using a Random Forest ML model.

## Project Structure

```
dyslexia_app/
├── app.py               ← Main Streamlit app (all pages)
├── requirements.txt     ← Python dependencies
├── data/
│   ├── labeled_dysx.csv    ← Training dataset
│   └── unlabeled_dysx.csv  ← Unlabeled dataset
└── README.md
```

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview of dyslexia, signs, causes, and how the tool works |
| 📋 Survey | 20-question parent/teacher observation survey |
| 🎯 Quiz | 10-question cognitive quiz for students |
| 🔬 Predict | ML model prediction with auto-filled quiz/survey scores |
| ℹ️ About | Dataset info, model details, and publications |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure the `data/` folder with `labeled_dysx.csv` is in the same directory as `app.py`.

## Deploying to Streamlit Community Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Connect your GitHub repo
4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
5. Click **Deploy** — your app will be live in ~2 minutes!

> The CSV files in `data/` must be committed to GitHub for the model to load.

## Deploying to Other Platforms

### Render
1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Railway
1. Connect repo at [railway.app](https://railway.app)
2. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

**Author**: Madhu Priya Pulletikurthi  
**Patent**: India No. 202341084582 A  
**© 2025 Dyslexia Detector**
