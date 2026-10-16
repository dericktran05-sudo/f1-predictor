import pandas as pd
import streamlit as st

preds = pd.read_parquet("predictions.parquet")
res = pd.read_parquet("race_results.parquet").sort_values("round")

st.title("F1 Race Outcome Predictor")
st.caption("2025 season. Model trained on 2023-2024 only.")

labels = {f"Round {int(r['round'])}: {r['circuit']}": int(r["round"])
          for _, r in res.iterrows()}
choice = st.selectbox("Pick a race", list(labels.keys()))
rnd = labels[choice]

info = res[res["round"] == rnd].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("DNFs in race", int(info["dnfs"]))
c2.metric("Model accuracy", f"{info['model_rho']:.2f}")
c3.metric("Grid-baseline accuracy", f"{info['grid_rho']:.2f}")

race = preds[preds["round"] == rnd].sort_values("model_rank")
table = race[["driver", "team", "grid", "model_rank", "finish", "dnf"]].copy()
table.columns = ["Driver", "Team", "Started", "Predicted", "Actual", "DNF"]
for col in ["Started", "Predicted", "Actual"]:
    table[col] = table[col].astype(int)
st.dataframe(table, hide_index=True, width="stretch")

st.subheader("The big picture")
st.image("chart_chaos.png")
st.image("chart_dnfs.png")