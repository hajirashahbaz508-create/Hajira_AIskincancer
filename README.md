# Skin Cancer Detection - Streamlit app

Files:
- app.py            : Streamlit application
- best_model.keras  : trained model
- model_info.json   : preprocessing + class info
- requirements.txt  : python dependencies (same versions as training)

Run locally:   pip install -r requirements.txt && streamlit run app.py

Deploy on Streamlit Community Cloud:
1. Create a GitHub repo and upload all files in this folder (use Git LFS if best_model.keras > 100 MB).
2. Go to https://share.streamlit.io -> New app -> select repo -> main file path: app.py
3. In Advanced settings choose Python 3.11 or 3.12 -> Deploy.
