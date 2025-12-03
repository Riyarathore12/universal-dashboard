import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Universal Data Analytics Dashboard", layout="centered")
st.title("📊 Universal Data Analytics Dashboard")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -------------------------
# Cached Functions
# -------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def compute_pivot(df, row, col, val, agg):
    try:
        pivot = pd.pivot_table(df, index=row, columns=col, values=val, aggfunc=agg)
        return pivot
    except:
        return None

# -------------------------
# Main Logic
# -------------------------
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Dataset uploaded successfully!")

    # -------------------------
    # Dataset Preview & Info
    # -------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    st.write("### Missing Values")
    st.dataframe(df.isna().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # -------------------------
    # Data Cleaning
    # -------------------------
    st.subheader("🧹 Data Cleaning Options")
    if st.checkbox("Remove Duplicate Rows"):
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.success(f"Removed {before - after} duplicate rows.")

    if st.checkbox("Fill Missing Values"):
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Missing values filled (numeric: median, categorical: mode).")

    st.write("### Cleaned Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # Visualizations
    # -------------------------
    st.subheader("📈 Visualizations")
    chart_type = st.selectbox(
        "Choose chart type",
        ["Histogram", "Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot", "Correlation Heatmap"]
    )

    col_x = col_y = None

    if chart_type in ["Histogram", "Line Chart", "Scatter Plot"]:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for this chart.")
        else:
            col_x = st.selectbox("X Column", numeric_cols)
            if chart_type in ["Line Chart", "Scatter Plot"]:
                col_y = st.selectbox("Y Column", numeric_cols)

    elif chart_type in ["Bar Chart", "Pie Chart"]:
        col_x = st.selectbox("Column", df.columns)

    fig, ax = plt.subplots(figsize=(5,4))
    plt.tight_layout()

    try:
        if chart_type == "Histogram" and col_x:
            colors = sns.color_palette("pastel", 10)
            ax.hist(df[col_x].dropna(), bins=20, color=colors[0], edgecolor='black')
            ax.set_xlabel(col_x)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {col_x}", fontsize=10)
            st.pyplot(fig)

        elif chart_type == "Bar Chart" and col_x:
            counts = df[col_x].value_counts().head(15)
            colors = sns.color_palette("bright", len(counts))
            counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_ylabel("Count")
            ax.set_title(f"Bar Chart of {col_x}", fontsize=10)
            st.pyplot(fig)

        elif chart_type == "Pie Chart" and col_x:
            counts = df[col_x].value_counts().head(10)
            colors = sns.color_palette("Set2", len(counts))
            counts.plot(kind='pie', autopct="%1.1f%%", ax=ax,
                        textprops={"fontsize": 8}, colors=colors)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {col_x}", fontsize=10)
            st.pyplot(fig)

        elif chart_type == "Line Chart" and col_x and col_y:
            ax.plot(df[col_x], df[col_y], marker='o', linestyle='-', color='green')
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"Line Chart: {col_x} vs {col_y}", fontsize=10)
            st.pyplot(fig)

        elif chart_type == "Scatter Plot" and col_x and col_y:
            ax.scatter(df[col_x], df[col_y], color='red', s=20, alpha=0.7)
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}", fontsize=10)
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax, cbar=True)
                ax.set_title("Correlation Heatmap", fontsize=10)
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for heatmap.")

    except Exception as e:
        st.error(f"Unable to create chart: {e}")

    # -------------------------
    # Pivot Table
    # -------------------------
    st.subheader("📊 Pivot Table")
    if len(numeric_cols) > 0 and len(df.columns) >= 2:
        row_pt = st.selectbox("Row", df.columns, key="row_pt")
        col_pt = st.selectbox("Column", df.columns, key="col_pt")
        val_pt = st.selectbox("Values", numeric_cols, key="val_pt")
        agg = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"], key="agg_func")

        pivot = compute_pivot(df, row_pt, col_pt, val_pt, agg)
        if pivot is not None:
            st.dataframe(pivot)
        else:
            st.warning("Pivot table cannot be created with selected columns.")
    else:
        st.warning("Not enough numeric columns for pivot table.")

else:
    st.info("👆 Please upload a CSV file to start.")
