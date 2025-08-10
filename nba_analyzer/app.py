# nba_analyzer/app.py

from pathlib import Path
import io
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from plotly.colors import qualitative as q

st.set_page_config(page_title="NBA Analytics", layout="wide")

# ---------- Paths ----------
DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------- Loaders ----------
@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)  

def read_uploaded(file, sheet=None) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, sheet_name=sheet)
    if name.endswith(".json"):
        return pd.read_json(file)
    if name.endswith(".txt"):
        content = file.read().decode("utf-8")
        try:
            return pd.read_csv(io.StringIO(content), sep=None, engine="python")
        except Exception:
            return pd.DataFrame({"content": content.splitlines()})
    raise ValueError(f"Unsupported file type: {name}")

CATEGORY_COLS = ["Player", "Team", "Pos", "Year"]
METRIC_COLS = [
    "G","GS","MP","FG","FGA","FG%","3P","3PA","3P%","2P","2PA","2P%","eFG%",
    "FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","Salary","Age"
]

def build_trend_ui(df: pd.DataFrame):

    # ---------- Row 1: Year slider ----------
    yrs = pd.to_numeric(df["Year"], errors="coerce").dropna()
    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    years = st.slider("", yr_min, yr_max, (yr_min, yr_max), label_visibility='collapsed')

    # ---------- Row 2: Team / Pos ----------
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        teams = st.multiselect("Team(s)", sorted(df["Team"].dropna().unique().tolist()))
    with r2c2:
        pos = st.multiselect("Pos", sorted(df["Pos"].dropna().unique().tolist()))

    # ---------- Row 3: Player / Metrics ----------
    r3c1, r3c2 = st.columns([1, 1])
    with r3c1:
        players = st.multiselect(
            "Player(s)",
            sorted(df["Player"].dropna().unique().tolist()),
            default=["LeBron James", "Stephen Curry", "Nikola Jokic"]
        )

    with r3c2:
        sel_metrics = st.multiselect(
            "Metric(s)",
            METRIC_COLS,
            default=["PTS"]
        )

    # ---------- Row 4: Aggregation controls ----------
    # Only offer "Aggregate On" among categories that currently have any selection,
    # falling back to all if nothing is selected yet.
    selected_cats = []
    if teams:   selected_cats.append("Team")
    if pos:     selected_cats.append("Pos")
    if players: selected_cats.append("Player")
    agg_candidates = selected_cats or ["Player", "Team", "Pos"]

    r4c1, r4c2 = st.columns([1, 1])
    with r4c1:
        aggregate_on = st.selectbox(
            "Aggregate On",
            agg_candidates,
            help="Choose which category becomes the legend and grouping dimension."
        )
    with r4c2:
        aggregate_by = st.selectbox(
            "Aggregate By",
            ["sum", "average"],
            help="Aggregate selected metrics across the other categories."
        )

    # ---------- Filtering ----------
    f = df.copy()
    if teams:   f = f[f["Team"].isin(teams)]
    if pos:     f = f[f["Pos"].isin(pos)]
    if players: f = f[f["Player"].isin(players)]
    f = f[(pd.to_numeric(f["Year"], errors="coerce") >= years[0]) &
          (pd.to_numeric(f["Year"], errors="coerce") <= years[1])]

    if not sel_metrics:
        st.info("Select at least one metric (right panel).")
        return
    if f.empty:
        st.warning("No data for the chosen filters.")
        return

    sel_metrics = [m for m in sel_metrics if m in f.columns]
    if not sel_metrics:
        st.warning("Selected metrics are not present in the dataset.")
        return

    # keep needed cols; cast numerics
    keep_cols = ["Year", "Player", "Team", "Pos"] + sel_metrics
    f = f[keep_cols].copy()
    f["Year"] = pd.to_numeric(f["Year"], errors="coerce").astype("Int64")
    for m in sel_metrics:
        f[m] = pd.to_numeric(f[m], errors="coerce")

    # ---------- Aggregate to the chosen category ----------
    # Group by Year and the chosen aggregate category; aggregate metrics across the rest.
    agg_func = np.sum if aggregate_by == "sum" else np.mean
    g = f.groupby(["Year", aggregate_on], dropna=False)[sel_metrics].agg(agg_func).reset_index()

    # ---------- Long format + combined legend ----------
    long = g.melt(
        id_vars=["Year", aggregate_on],
        value_vars=sel_metrics,
        var_name="Metric",
        value_name="Value"
    )
    # Combined legend label, but keep color consistent per aggregate value
    long["Legend"] = long[aggregate_on].astype(str) + " \u2022 " + long["Metric"]
    long["ColorKey"] = long[aggregate_on].astype(str)

    # ---------- Colors: one color per aggregate value (metrics vary by dash) ----------

    # First group: blues & oranges (from Plotly + D3)
    # --- Custom 18-color palette ---
    first_group = [
        "#1f77b4",  # Medium blue
        "#08306b",  # Very dark navy
        "#6baed6",  # Light sky blue
        "#ff7f0e",  # Bright orange
        "#a63603",  # Dark burnt orange
        "#fdae6b"   # Light orange
    ]
    
    second_group = [
        "#54278f",  # Deep violet
        "#756bb1",  # Medium violet
        "#dadaeb",  # Light lavender
        "#b8860b",  # Dark goldenrod
        "#ffd92f",  # Bright yellow
        "#ffe87c"   # Light yellow
    ]
    
    third_group = [
        "#00441b",  # Deep forest green
        "#238b45",  # Medium green
        "#a1d99b",  # Light green
        "#7f7f7f",  # Medium gray
        "#bdbdbd",  # Light gray
        "#f0f0f0"   # Near white
    ]
    
    palette = first_group + second_group + third_group

    entities = sorted(long["ColorKey"].dropna().unique())
    entity_color = {e: palette[i % len(palette)] for i, e in enumerate(entities)}
    # Map each Legend entry to its entity's color
    color_map = {}
    for _, row in long[["Legend", "ColorKey"]].drop_duplicates().iterrows():
        color_map[row["Legend"]] = entity_color.get(row["ColorKey"], None)

    # ---------- Plot ----------
    fig = px.line(
        long,
        x="Year",
        y="Value",
        color="Legend",          # combined legend (AggregateValue • Metric)
        line_dash="Metric",      # same entity uses same color; metric differs by dash
        markers=True,
        hover_data=[aggregate_on, "Metric"]
    )
    # Remove Plotly's automatic ", Metric" suffix from legend labels
    fig.for_each_trace(lambda t: t.update(name=t.name.split(",")[0]))

    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        height=500,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        coloraxis_showscale=False
    )
    # Apply color map (only for the traces we have)
    fig.for_each_trace(lambda t: t.update(line=dict(color=color_map.get(t.name, None))))
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True)


# ---------- Summary ----------
def summarize_dataframe(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True)

    st.write("Summary Statistics")
    dupes = df.duplicated().sum()
    st.code(f"Shape: {(df.shape[0], df.shape[1])}, Duplicate Rows: {dupes}")
    st.code("Columns: " + ", ".join(map(str, df.columns)))

    rows = []
    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        non_null = s.notnull().sum()
        nulls = s.isnull().sum()
        uniq = s.nunique()

        total = mean = minv = maxv = modev = ""
        if pd.api.types.is_numeric_dtype(s):
            total = s.sum()
            mean = s.mean()
            minv = s.min()
            maxv = s.max()
            modev = s.mode().iloc[0] if not s.mode().empty else ""
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            modev = s.mode().iloc[0] if not s.mode().empty else ""

        rows.append([col, dtype, non_null, nulls, uniq, total, mean, modev, minv, maxv])

    summary_df = pd.DataFrame(
        rows,
        columns=["Column", "Type", "Non-Null", "Null Count", "Unique",
                 "Total", "Mean", "Mode", "Min", "Max"]
    )
    st.dataframe(summary_df, use_container_width=True)

# ---------- Sidebar: choose data source ----------
st.sidebar.header("NBA Analyzer")
choice = st.sidebar.radio("Choose:", ["Built-in dataset", "Upload file"], index=0)

df = None
error = None

if choice == "Built-in dataset":
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        st.sidebar.warning("No .parquet files found in /data.")
    else:
        pick = st.sidebar.selectbox("Built-in .parquet", parquet_files, format_func=lambda p: p.name)
        try:
            df = load_parquet(pick)
        except Exception as e:
            error = str(e)
else:
    up = st.sidebar.file_uploader("Upload CSV / Excel / JSON / TXT",
                                  type=["csv", "xlsx", "xls", "json", "txt"])
    sheet = None
    if up and up.name.lower().endswith((".xlsx", ".xls")):
        # peek sheets
        xls = pd.ExcelFile(up)
        sheet = st.sidebar.selectbox("Excel sheet", xls.sheet_names, index=0)
        up.seek(0)  # reset after peek
    if up:
        try:
            df = read_uploaded(up, sheet=sheet)
        except Exception as e:
            error = str(e)

# ---------- Main ----------
if error:
    st.error(error)
elif df is None:
    st.info("Select a built-in dataset or upload a file in the sidebar to begin.")
else:
    
    # Show trends
    st.write("Analytics Dashboard")
    build_trend_ui(df)

    # Then the preview + stats
    st.write("Dataset Overview")
    summarize_dataframe(df)
