import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Race Outcome Predictor", layout="wide")
st.title("F1 Race Outcome Predictor")


def race_summary(df, model_col):
    rows = []
    for rnd, race in df.groupby("round"):
        rows.append({
            "round": int(rnd),
            "circuit": race["circuit"].iloc[0],
            "dnfs": int(race["dnf"].sum()),
            "model_rho": race[model_col].corr(race["finish"], method="spearman"),
            "grid_rho": race["grid_rank"].corr(race["finish"], method="spearman"),
        })
    return pd.DataFrame(rows).sort_values("round")


def show_season(df, model_col, key):
    summary = race_summary(df, model_col)

    c1, c2, c3 = st.columns(3)
    c1.metric("Races", len(summary))
    c2.metric("Model (avg Spearman)", f"{summary['model_rho'].mean():.3f}")
    c3.metric("Grid baseline (avg Spearman)", f"{summary['grid_rho'].mean():.3f}")

    st.line_chart(summary.set_index("round")[["model_rho", "grid_rho"]])
    st.caption("Accuracy per race (1.0 = perfect order). The model and the grid baseline track each other closely.")

    labels = {f"Round {int(r['round'])}: {r['circuit']}": int(r["round"])
              for _, r in summary.iterrows()}
    choice = st.selectbox("Pick a race", list(labels.keys()), key=key)
    rnd = labels[choice]

    info = summary[summary["round"] == rnd].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("DNFs in race", int(info["dnfs"]))
    m2.metric("Model accuracy", f"{info['model_rho']:.2f}")
    m3.metric("Grid-baseline accuracy", f"{info['grid_rho']:.2f}")

    race = df[df["round"] == rnd].sort_values(model_col)
    table = race[["driver", "team", "grid", model_col, "finish", "dnf"]].copy()
    table.columns = ["Driver", "Team", "Started", "Predicted", "Actual", "DNF"]
    for col in ["Started", "Predicted", "Actual"]:
        table[col] = table[col].astype(int)
    st.dataframe(table, hide_index=True, width="stretch")


tab26, tab25, tab_next = st.tabs(["2026 season", "2025 test season", "Next race"])

with tab26:
    st.caption("2026: new rules and 11 teams. The model is retrained before every race on all earlier races.")
    show_season(pd.read_parquet("predictions_2026.parquet"), "rank_walk", "sel26")

with tab25:
    st.caption("2025: model trained on 2023-2024 only.")
    show_season(pd.read_parquet("predictions.parquet"), "model_rank", "sel25")
    st.subheader("The big picture")
    st.image("chart_chaos.png")
    st.image("chart_dnfs.png")

with tab_next:
    files = glob.glob("prediction_round_*.csv")
    if not files:
        st.info("No prediction saved yet. One is made after each qualifying session.")
    else:
        latest = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        rnd = int(latest.split("_")[-1].split(".")[0])
        st.write(f"Latest saved prediction: round {rnd}")
        st.caption("Made from the qualifying result, before the race. Grid penalties are not included.")
        pred = pd.read_csv(latest)
        pred.columns = ["Predicted", "Driver", "Team", "Started (qualifying)"]
        st.dataframe(pred, hide_index=True, width="stretch")