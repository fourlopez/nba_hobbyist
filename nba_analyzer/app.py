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

# CATEGORY_COLS = ["Player", "Team", "Pos", "Year"]
# METRIC_COLS = [
#     "G","GS","MP","FG","FGA","FG%","3P","3PA","3P%","2P","2PA","2P%","eFG%",
#     "FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","Salary","Age"
# ]
def _is_numeric_like(series: pd.Series, min_ratio: float = 0.85) -> bool:
    # Treat strings like "1,234", "98.7%", "  42 " as numeric-like
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    coerced = pd.to_numeric(s, errors="coerce")
    valid_ratio = coerced.notna().mean()
    return valid_ratio >= min_ratio

def infer_schema(df: pd.DataFrame, cat_max_unique: int = 40):
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    likely_year = [c for c in cols if lower[c] in ("year", "yr")]
    dt_candidates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]

    # year-like numeric detection (same as your version) ...
    year_like_numeric = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                s_non = s.dropna()
                if (s_non.between(1900, 2100)).mean() > 0.8 and s_non.nunique() >= 3:
                    year_like_numeric.append(c)

    # choose year_col (same priority)
    if likely_year:
        year_col = likely_year[0]
    elif dt_candidates:
        year_col = dt_candidates[0]
    elif year_like_numeric:
        year_col = year_like_numeric[0]
    else:
        any_num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        year_col = any_num[0] if any_num else cols[0]

    # --- Enhanced metric detection ---
    metric_candidates = []
    for c in cols:
        if c == year_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            metric_candidates.append(c)
        elif pd.api.types.is_object_dtype(s):
            if _is_numeric_like(s):
                metric_candidates.append(c)

    # Categories: objects/categoricals/booleans OR low-card numerics NOT claimed as metrics
    category_candidates = []
    for c in cols:
        if c in (year_col,):
            continue
        if c in metric_candidates:
            continue
        s = df[c]
        if (pd.api.types.is_object_dtype(s) or
            pd.api.types.is_categorical_dtype(s) or
            pd.api.types.is_bool_dtype(s)):
            category_candidates.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            nun = s.nunique(dropna=True)
            if 1 < nun <= cat_max_unique:
                category_candidates.append(c)

    priority = ["Player", "Team", "Pos"]
    category_candidates = sorted(category_candidates, key=lambda c: (c not in priority, c))
    return year_col, category_candidates, metric_candidates



def build_trend_ui(df: pd.DataFrame):

    # ---------- Infer roles; allow overrides ----------
    infer_year, infer_cats, infer_metrics = infer_schema(df)

    with st.expander("Schema Settings", expanded=False):
        year_col = st.selectbox("X-axis column", [infer_year] + [c for c in df.columns if c != infer_year], index=0)
        all_cols = list(df.columns)
        cat_cols = st.multiselect("Category columns", all_cols, default=infer_cats)
        metric_cols = st.multiselect("Metric columns", [c for c in all_cols if c != year_col],
                                     default=infer_metrics or [])

    if not metric_cols:
        st.info("Select at least one metric column in the Schema expander.")
        return
    if not cat_cols:
        st.info("Select at least one category column in the Schema expander.")
        return

    # Cast year/x
    f = df.copy()
    if pd.api.types.is_datetime64_any_dtype(f[year_col]):
        # ok for slider with datetimes
        x_series = f[year_col]
        x_min, x_max = x_series.min(), x_series.max()
        years = st.slider("", x_min, x_max, (x_min, x_max), label_visibility='collapsed')
        f = f[(f[year_col] >= years[0]) & (f[year_col] <= years[1])]
    else:
        # try numeric
        x_num = pd.to_numeric(f[year_col], errors="coerce")
        f[year_col] = x_num
        yrs = x_num.dropna()
        if yrs.empty:
            st.warning(f"No numeric values for x-axis column '{year_col}'.")
            return
        yr_min, yr_max = int(yrs.min()), int(yrs.max())
        years = st.slider("", yr_min, yr_max, (yr_min, yr_max), label_visibility='collapsed')
        f = f[(pd.to_numeric(f[year_col], errors="coerce") >= years[0]) &
              (pd.to_numeric(f[year_col], errors="coerce") <= years[1])]

    # ---------- Dynamic category filters ----------
    # One multiselect per category column (like your current Team/Pos/Player)
    cat_selections = {}
    cat_cols_layout = st.columns(min(4, len(cat_cols))) if len(cat_cols) > 1 else [st]
    for i, c in enumerate(cat_cols):
        with cat_cols_layout[i % len(cat_cols_layout)]:
            opts = sorted(pd.Series(f[c]).dropna().unique().tolist())
            cat_selections[c] = st.multiselect(c, opts)

    # Apply category filters
    for c, vals in cat_selections.items():
        if vals:
            f = f[f[c].isin(vals)]

    # ---------- Metric selection (single control) ----------
    sel_metrics = st.multiselect("Metric(s)", metric_cols, default=[metric_cols[0]])

    if not sel_metrics:
        st.info("Select at least one metric.")
        return
    if f.empty:
        st.warning("No data for the chosen filters.")
        return

    # ---------- Aggregation controls ----------
    # Only allow Aggregate On among the chosen category columns; prefer ones with any selection
    selected_cats = [c for c, v in cat_selections.items() if v]
    agg_candidates = selected_cats or cat_cols
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        aggregate_on = st.selectbox("Aggregate On", agg_candidates,
                                    help="Legend/grouping category. Metrics are aggregated across other categories.")
    with r4c2:
        aggregate_by = st.selectbox("Aggregate By", ["sum", "average"])

    # ---------- Type cleanup ----------
    for m in sel_metrics:
        f[m] = pd.to_numeric(f[m], errors="coerce")

    # ---------- Aggregate to chosen category ----------
    other_dims = [c for c in cat_cols if c != aggregate_on]
    agg_func = np.sum if aggregate_by == "sum" else np.mean
    # Group only by x and aggregate_on; this collapses over 'other_dims'
    g = f.groupby([year_col, aggregate_on], dropna=False)[sel_metrics].agg(agg_func).reset_index()

    # ---------- Long format + legend ----------
    long = g.melt(
        id_vars=[year_col, aggregate_on],
        value_vars=sel_metrics,
        var_name="Metric",
        value_name="Value"
    )
    long["Legend"] = long[aggregate_on].astype(str) + " \u2022 " + long["Metric"]
    long["ColorKey"] = long[aggregate_on].astype(str)

    # ---------- Palette (your alternating 18-color set) ----------
    first_group = ["#1f77b4","#ff7f0e","#08306b","#a63603","#6baed6","#fdae6b"]
    second_group = ["#54278f","#ffd92f","#756bb1","#b8860b","#dadaeb","#ffe87c"]
    third_group  = ["#00441b","#7f7f7f","#238b45","#bdbdbd","#a1d99b","#f0f0f0"]
    palette = first_group + second_group + third_group

    entities = sorted(long["ColorKey"].dropna().unique())
    entity_color = {e: palette[i % len(palette)] for i, e in enumerate(entities)}
    color_map = {row["Legend"]: entity_color.get(row["ColorKey"], None)
                 for _, row in long[["Legend","ColorKey"]].drop_duplicates().iterrows()}

    # ---------- Plot ----------
    fig = px.line(
        long,
        x=year_col,
        y="Value",
        color="Legend",
        line_dash="Metric",
        markers=True,
        hover_data=[aggregate_on, "Metric"]
    )
    # Strip duplicate metric suffix that Plotly adds
    fig.for_each_trace(lambda t: t.update(name=t.name.split(",")[0]))
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        height=500,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        coloraxis_showscale=False
    )
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
# ---------- Main ----------
if error:
    st.error(error)
elif df is None:
    st.info("Select a built-in dataset or upload a file in the sidebar to begin.")
else:
    st.write("Analytics Dashboard")
    
    try:
        build_trend_ui(df)   # graphing & filters
    except Exception as e:
        st.error(f"Error in Analytics Dashboard: {e}")

    # This will run regardless of graphing errors
    st.write("Dataset Overview")
    summarize_dataframe(df)
