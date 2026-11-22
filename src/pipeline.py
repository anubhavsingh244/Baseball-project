"""Model MLB team wins from same-season stats and build a static Plotly site.

Run:  python src/pipeline.py
Output: site/index.html and site/metrics.json
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Teams.csv"
SITE_DIR = ROOT / "site"

# Baseball Databank (Chadwick Baseball Bureau), CC BY-SA 3.0.
# Pinned to a commit of the xorq-labs mirror so the same code gives the same results.
DATA_COMMIT = "7c6d48bae384c0b2669d0ee1133b99ed083dace6"
DATA_URL = (
    f"https://raw.githubusercontent.com/xorq-labs/baseballdatabank/{DATA_COMMIT}/core/Teams.csv"
)
DOWNLOAD_TIMEOUT_S = 30

FIRST_YEAR = 1900
TRAIN_BEFORE = 2005  # seasons before this train the model; this year onward test it
GAMES = 162
PYTHAG_EXPONENT = 1.83

COUNT_COLS = ["R", "RA", "H", "HR", "BB", "SO", "HRA", "BBA", "SOA", "E"]
FEATURES = [f"{c}_pg" for c in COUNT_COLS] + ["ERA"]
REQUIRED_COLS = set(COUNT_COLS) | {"yearID", "G", "W", "L", "ERA", "name"}


def download(url: str) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": "baseball-project"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
        return pd.read_csv(io.BytesIO(response.read()))


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            cached = pd.read_csv(DATA_PATH)
            if REQUIRED_COLS.issubset(cached.columns):
                return cached
        except Exception:  # unreadable cache, download again
            pass

    df = download(DATA_URL)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Teams.csv is missing columns: {sorted(missing)}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(DATA_PATH)  # atomic, so a killed run never leaves a partial cache
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["yearID"] >= FIRST_YEAR].copy()
    df = df.dropna(subset=COUNT_COLS + ["ERA", "W", "L", "G"])
    df = df[(df["G"] > 0) & ((df["W"] + df["L"]) > 0) & ((df["R"] + df["RA"]) > 0)]
    for c in COUNT_COLS:
        df[f"{c}_pg"] = df[c] / df["G"]
    df["win_pct"] = df["W"] / (df["W"] + df["L"])
    runs = df["R"] ** PYTHAG_EXPONENT
    runs_allowed = df["RA"] ** PYTHAG_EXPONENT
    df["pythag"] = runs / (runs + runs_allowed)
    df["label"] = df["name"] + " " + df["yearID"].astype(str)
    return df.reset_index(drop=True)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae_wins": float(mean_absolute_error(y_true, y_pred) * GAMES),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(df: pd.DataFrame):
    train_df = df[df["yearID"] < TRAIN_BEFORE]
    test_df = df[df["yearID"] >= TRAIN_BEFORE].copy()
    if len(train_df) < 100 or len(test_df) < 100:
        raise RuntimeError(f"Unexpected split sizes: train={len(train_df)}, test={len(test_df)}")

    x_train, y_train = train_df[FEATURES], train_df["win_pct"]
    x_test, y_test = test_df[FEATURES], test_df["win_pct"]

    linear = LinearRegression().fit(x_train, y_train)
    forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
    ).fit(x_train, y_train)

    test_df["pred_linear"] = linear.predict(x_test)
    test_df["pred_forest"] = forest.predict(x_test)

    metrics = {
        "Pythagorean baseline": evaluate(y_test, test_df["pythag"].to_numpy()),
        "Linear regression": evaluate(y_test, test_df["pred_linear"].to_numpy()),
        "Random forest": evaluate(y_test, test_df["pred_forest"].to_numpy()),
    }
    importances = pd.Series(forest.feature_importances_, index=FEATURES).sort_values()
    return test_df, metrics, importances, len(train_df), len(test_df)


def fig_actual_vs_predicted(test_df: pd.DataFrame, best_col: str, best_name: str) -> go.Figure:
    actual = test_df["win_pct"] * GAMES
    pred = test_df[best_col] * GAMES
    lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual, y=pred, mode="markers", name="Team season",
            text=test_df["label"],
            hovertemplate="%{text}<br>Actual: %{x:.1f}<br>Estimated: %{y:.1f}<extra></extra>",
            marker=dict(size=7, opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect match",
                   line=dict(dash="dash", color="gray"))
    )
    fig.update_layout(
        title=f"Actual vs estimated wins per {GAMES} games (lowest test MAE: {best_name}, {TRAIN_BEFORE}+ test seasons)",
        xaxis_title="Actual wins (per 162 games)",
        yaxis_title="Estimated wins (per 162 games)",
    )
    return fig


def fig_model_comparison(metrics: dict) -> go.Figure:
    names = list(metrics)
    fig = go.Figure(
        go.Bar(
            x=names, y=[metrics[n]["mae_wins"] for n in names],
            text=[f"{metrics[n]['mae_wins']:.2f}" for n in names], textposition="outside",
        )
    )
    fig.update_layout(
        title="Test error by model (lower is better)",
        yaxis_title="Mean absolute error (wins per 162 games)",
    )
    return fig


def fig_importance(importances: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h"))
    fig.update_layout(
        title="Random forest feature importance",
        xaxis_title="Importance",
        yaxis_title="Feature (per game; ERA is per 9 innings)",
    )
    return fig


def fig_trends(df: pd.DataFrame) -> go.Figure:
    by_year = df.groupby("yearID").agg(R=("R", "sum"), HR=("HR", "sum"), G=("G", "sum"))
    # each game is counted once per team, so runs per team-game is the per-team average
    by_year["runs_pg"] = by_year["R"] / by_year["G"]
    by_year["hr_pg"] = by_year["HR"] / by_year["G"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=by_year.index, y=by_year["runs_pg"], name="Runs per team game"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=by_year.index, y=by_year["hr_pg"], name="HR per team game"),
                  secondary_y=True)
    fig.update_layout(title="League scoring and home runs over time", xaxis_title="Season")
    fig.update_yaxes(title_text="Runs per team game", secondary_y=False)
    fig.update_yaxes(title_text="HR per team game", secondary_y=True)
    return fig


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What explains MLB team wins?</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ccc; padding: .4rem .8rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .chart {{ margin: 2rem 0; }}
 footer {{ font-size: .85rem; color: #666; margin-top: 3rem; }}
</style>
</head>
<body>
<h1>What explains MLB team wins?</h1>
<p>Each model estimates a team's win percentage from that same season's per-game batting, pitching and fielding stats.
This explains a season after the fact. It does not forecast future wins.
Models train on {train_n} team seasons from {first}-{train_end} and are tested on {test_n} seasons from {test_start} onward.</p>
<table>
<tr><th>Model</th><th>MAE (wins/162)</th><th>R&sup2;</th></tr>
{rows}
</table>
{charts}
<footer>Data: <a href="https://github.com/chadwickbureau/baseballdatabank">Baseball Databank</a>
(Chadwick Baseball Bureau, CC BY-SA 3.0), read from a pinned copy in the
<a href="https://github.com/xorq-labs/baseballdatabank/tree/{data_commit}">xorq-labs mirror</a>.</footer>
</body>
</html>
"""


def build_site(figs: list[go.Figure], metrics: dict, n_train: int, n_test: int) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    charts = []
    for i, fig in enumerate(figs):
        html = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        charts.append(f'<div class="chart">{html}</div>')
    rows = "\n".join(
        f"<tr><td>{name}</td><td>{m['mae_wins']:.2f}</td><td>{m['r2']:.3f}</td></tr>"
        for name, m in metrics.items()
    )
    page = PAGE.format(
        train_n=n_train, test_n=n_test, first=FIRST_YEAR, train_end=TRAIN_BEFORE - 1,
        test_start=TRAIN_BEFORE, rows=rows, charts="\n".join(charts), data_commit=DATA_COMMIT,
    )
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")
    (SITE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    df = prepare(load_data())
    test_df, metrics, importances, n_train, n_test = train(df)

    model_cols = {
        "Pythagorean baseline": "pythag",
        "Linear regression": "pred_linear",
        "Random forest": "pred_forest",
    }
    best_name = min(model_cols, key=lambda n: metrics[n]["mae_wins"])

    figs = [
        fig_actual_vs_predicted(test_df, model_cols[best_name], best_name),
        fig_model_comparison(metrics),
        fig_importance(importances),
        fig_trends(df),
    ]
    build_site(figs, metrics, n_train, n_test)
    for name, m in metrics.items():
        print(f"{name}: MAE {m['mae_wins']:.2f} wins, R2 {m['r2']:.3f}")
    print(f"Wrote {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
