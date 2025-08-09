from pathlib import Path
import io
import pandas as pd
import streamlit as st

# --- locations ---
DATA_DIR = Path(__file__).resolve().parent / "data"

# --- cached loaders ---
@st.cache_data
def load_builtin_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)  # requires pyarrow

def read_uploaded(file, sheet=None):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, sheet_name=sheet)
    elif name.endswith(".json"):
        return pd.read_json(file)
    elif name.endswith(".txt"):
        content = file.read().decode("utf-8")
        try:
            return pd.read_csv(io.StringIO(content), sep=None, engine="python")
        except Exception:
            return pd.DataFrame({"content": content.splitlines()})
    else:
        raise ValueError(f"Unsupported file type: {name}")

# === SIDEBAR: choose data source ===
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose:", ["Built-in dataset", "Upload file"], index=0)

df = None
error = None

if source == "Built-in dataset":
    # discover available .parquet files in data/
    parquet_files = sorted([p for p in DATA_DIR.glob("*.parquet")])
    if not parquet_files:
        st.sidebar.warning("No .parquet files found in /data. Switch to Upload.")
    else:
        choice = st.sidebar.selectbox(
            "Built-in .parquet",
            options=parquet_files,
            format_func=lambda p: p.name,
        )
        try:
            df = load_builtin_parquet(choice)
        except Exception as e:
            error = str(e)

else:  # Upload file
    file = st.sidebar.file_uploader("Upload CSV / Excel / JSON / TXT",
                                    type=["csv", "xlsx", "xls", "json", "txt"])
    sheet = None
    if file and file.name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(file)
        sheet = st.sidebar.selectbox("Excel sheet", xls.sheet_names, index=0)
        file.seek(0)
    if file:
        try:
            df = read_uploaded(file, sheet=sheet)
        except Exception as e:
            error = str(e)

# === MAIN ===
st.title("NBA Analyzer")

if error:
    st.error(error)
elif df is not None:
    # call your existing summary block
    summarize_dataframe(df)  # <-- keep your previous function
else:
    st.info("Select a built-in dataset or upload a file in the sidebar to begin.")
