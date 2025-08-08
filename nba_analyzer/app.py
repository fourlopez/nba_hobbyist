import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="NBA Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("NBA Analyzer")

uploaded_file = st.sidebar.file_uploader(
    "Upload file (CSV, Excel, JSON, TXT)",
    type=["csv", "xlsx", "xls", "json", "txt"]
)

def load_file(uploaded_file):
    df, error = None, None
    if uploaded_file:
        try:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif ext == "json":
                df = pd.read_json(uploaded_file)
            elif ext == "txt":
                content = uploaded_file.read().decode("utf-8")
                try:
                    df = pd.read_csv(io.StringIO(content), sep=None, engine="python")
                except Exception:
                    df = pd.DataFrame({"content": content.splitlines()})
            else:
                error = f"Unsupported file extension: {ext}"
        except Exception as e:
            error = str(e)
    return df, error

def summarize_dataframe(df):
    st.subheader("Preview")
    st.dataframe(df)

    st.subheader("Dataset Summary")
    duplicate_count = df.duplicated().sum()
    st.code(f"Shape: {(df.shape[0], df.shape[1])}, Duplicate Rows: {duplicate_count}")
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

    summary_df = pd.DataFrame(rows, columns=[
        "Column", "Type", "Non-Null", "Null Count", "Unique", "Total", "Mean", "Mode", "Min", "Max"
    ])
    st.subheader("Column Summary")
    st.dataframe(summary_df, use_container_width=True)

if uploaded_file:
    df, err = load_file(uploaded_file)
    if err:
        st.error(f"Error: {err}")
    elif df is not None:
        summarize_dataframe(df)
else:
    st.info("Upload a CSV, Excel, JSON, or TXT file to begin.")
