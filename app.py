import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ART = "artifacts"

@st.cache_resource
def load_model():
    return joblib.load(f"{ART}/abx_ranker.pkl")

@st.cache_data
def load_data():
    train = pd.read_parquet(f"{ART}/train_agg.parquet")
    all_abx = json.load(open(f"{ART}/all_abx.json"))
    return train, all_abx

model = load_model()
train, all_abx = load_data()

cat_cols = ["Genome ID", "Antibiotic", "mode_mtype", "mode_msign"]
num_cols = ["median_log_mvalue"]

lookup = train.set_index(["Genome ID", "Antibiotic"])
genomes = sorted(train["Genome ID"].astype(str).unique())

def recommend_top5(genome_id: str, topk: int = 5) -> pd.DataFrame:
    genome_id = str(genome_id).strip()

    rows = []
    for abx in all_abx:
        key = (genome_id, abx)
        if key in lookup.index:
            r = lookup.loc[key]
            rows.append([genome_id, abx, r["mode_mtype"], r["mode_msign"], r["median_log_mvalue"]])
        else:
            rows.append([genome_id, abx, "unknown", "=", np.nan])

    cand = pd.DataFrame(rows, columns=["Genome ID", "Antibiotic", "mode_mtype", "mode_msign", "median_log_mvalue"])
    p = model.predict_proba(cand[cat_cols + num_cols])[:, 1]

    out = cand[["Antibiotic"]].copy()
    out["P_effective"] = p
    out = out.sort_values("P_effective", ascending=False).head(topk)
    return out.reset_index(drop=True)

st.title("E. coli Antibiotic Recommender")

genome_id = st.selectbox("Choose a Genome ID", genomes)
topk = st.slider("Top N antibiotics", 1, 10, 5)

if st.button("Recommend"):
    rec = recommend_top5(genome_id, topk=topk)
    st.dataframe(rec)