# nba_analyzer/app.py

from pathlib import Path
import io
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px


st.set_page_config(page_title="NBA Analyzer", layout="wide")
# st.title("NBA Analyzer")

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

# ---------- Trends (compound line) ----------


CATEGORY_COLS = ["Player", "Team", "Pos", "Year"]
METRIC_COLS = [
    "G","GS","MP","FG","FGA","FG%","3P","3PA","3P%","2P","2PA","2P%","eFG%",
    "FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","Salary","Age"
]

def build_trend_ui(df: pd.DataFrame):
    st.subheader("Trends")

    # --- Filters ---
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        players = st.multiselect("Player(s)", sorted(df["Player"].dropna().unique().tolist()))
    with c2:
        teams = st.multiselect("Team(s)", sorted(df["Team"].dropna().unique().tolist()))
    with c3:
        pos = st.multiselect("Pos", sorted(df["Pos"].dropna().unique().tolist()))
    with c4:
        yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
        years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    # apply filters
    f = df.copy()
    if players: f = f[f["Player"].isin(players)]
    if teams:   f = f[f["Team"].isin(teams)]
    if pos:     f = f[f["Pos"].isin(pos)]
    f = f[(f["Year"] >= years[0]) & (f["Year"] <= years[1])]

    # --- Metrics & options ---
    m1, m2, m3, m4 = st.columns([1.4, 1, 1, 1])
    with m1:
        sel_metrics = st.multiselect("Metrics", METRIC_COLS, default=["PTS"])
    with m2:
        by_player = st.checkbox("One chart per player (facet)", value=len(players) > 1)
    with m3:
        normalize = st.checkbox("Normalize metrics", value=False,
                                help="Min-max per metric (0-1) so different scales compare.")
    with m4:
        ma = st.checkbox("Moving average", value=False)
    if ma:
        win = st.number_input("MA window", min_value=2, max_value=20, value=3, step=1)

    if not sel_metrics:
        st.info("Select at least one metric to draw.")
        return

    if f.empty:
        st.warning("No data for the chosen filters.")
        return

    # keep needed cols, cast Year to int
    f = f[CATEGORY_COLS + sel_metrics].copy()
    f["Year"] = f["Year"].astype(int)

    # melt to long for multi-metric compound lines
    long = f.melt(id_vars=["Year","Player","Team","Pos"],
                  value_vars=sel_metrics, var_name="Metric", value_name="Value")

    # normalize per metric (and optionally per player)
    if normalize:
        long["Value"] = long.groupby(["Metric"])["Value"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
        )

    # optional moving average per player+metric (and year-sorted)
    if ma:
        long = long.sort_values(["Player","Metric","Year"])
        long["Value"] = long.groupby(["Player","Metric"], dropna=False)["Value"] \
                            .transform(lambda s: s.rolling(win, min_periods=1).mean())

    # Build figure
    # color = Metric (compound lines); facet by Player if requested and >1 players
    facet = "Player" if by_player and long["Player"].nunique() > 1 else None

    fig = px.line(
        long,
        x="Year", y="Value",
        color="Metric",
        facet_col=facet,
        facet_col_wrap=2 if facet else None,
        markers=True
    )
    fig.update_layout(height=500 if not facet else 500 + 200 * (int(np.ceil(long["Player"].nunique()/2))-1),
                      legend_title_text="Metric")

    st.plotly_chart(fig, use_container_width=True)

# ---------- Summary ----------
def summarize_dataframe(df: pd.DataFrame):
    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Summary")
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
    st.subheader("Column Summary")
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
    # Show trends FIRST (top of page)
    st.header("Trends")
    build_trend_ui(df)

    st.divider()

    # Then the preview + stats
    st.header("Dataset Overview")
    summarize_dataframe(df)

