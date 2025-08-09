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
    return pd.read_parquet(path)  # requires pyarrow

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
    years = st.slider("", yr_min, yr_max, (yr_min, yr_max))

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
            default=["LeBron James"]
        )

    with r3c2:
        sel_metrics = st.multiselect(
            "Metric(s)",
            METRIC_COLS,
            default=["PTS", "AST", "TRB", "BLK", "STL"]
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
    f = f[CATEGORY_COLS + sel_metrics].copy()
    f["Year"] = pd.to_numeric(f["Year"], errors="coerce").astype("Int64")
    for m in sel_metrics:
        f[m] = pd.to_numeric(f[m], errors="coerce")

    # long format for compound lines (FIX: value_name should be "Value")
    long = f.melt(
        id_vars=["Year","Player","Team","Pos"],
        value_vars=sel_metrics,
        var_name="Metric",
        value_name="Value"
    )

    # distinct player colors (choose any qualitative palette or combine a few)
    from plotly.colors import qualitative as q
    palette = q.D3 + q.Set2 + q.Set3
    players_in_view = sorted(long["Player"].dropna().unique())
    color_map = {p: palette[i % len(palette)] for i, p in enumerate(players_in_view)}
    
    # single compound line chart
    fig = px.line(
        long,
        x="Year", y="Value",
        color="Player",                 # colors by player only
        line_dash="Metric",             # metrics use dash styles (not colors)
        markers=True,
        hover_data=["Player", "Metric"],
        color_discrete_map=color_map    # deterministic colors per player
    )
    fig.update_layout(
        height=500,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
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
