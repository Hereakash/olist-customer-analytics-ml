# olist-customer-analytics-ml

Streamlit dashboard + ML for Olist customer analytics:
- RFM segmentation, cohort retention, delivery performance, category sales, review trends
- AI/ML reorder prediction (RandomForest + PyTorch Neural Network)
- Batch CSV upload scoring
- Prediction logging with run history (`run_id`) and monitoring charts

## How to run (Windows)

### 1) Create & activate venv (optional)
```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bat
pip install -r requirements.txt
```

### 3) Configure DB (DO NOT commit real password)
Copy `.env.example` to `config/settings.env` and fill values:
```bat
copy .env.example config\settings.env
```

### 4) Run Streamlit
```bat
streamlit run dashboard\streamlit_app.py
```

## Notes
- Raw dataset files are not included in this repo (`data/` is ignored).
- `config/settings.env` contains secrets and is ignored by git.