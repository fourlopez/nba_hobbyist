# nba_analyzer/app.py

from pathlib import Path
import io
import pandas as pd
import streamlit as st

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
    summarize_dataframe(df)
