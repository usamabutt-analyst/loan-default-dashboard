"""
Clean • Viz • Insights - Production Streamlit Data Analysis App
A comprehensive tool for data upload, cleaning, visualization, and auto-generated insights.
"""

import io
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Clean • Viz • Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# === Sidebar Branding ===
st.sidebar.title("📊 Data Analytics App")
st.sidebar.markdown("---")
st.sidebar.markdown("👤 **Developed by Usama Butt**")  # Your Name Here
st.sidebar.markdown("---")

# === Main App ===
st.title("My Data Analytics Dashboard")

# Example: file uploader
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # File read
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully ✅")
        st.dataframe(df.head())  # Show first rows
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a file to continue.")


# -------------------- Session state init --------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df: Optional[pd.DataFrame] = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df: Optional[pd.DataFrame] = None

# -------------------- Helpers --------------------
def read_file(uploaded_file) -> pd.DataFrame:
    """
    Read uploaded CSV/XLSX and return DataFrame. Tries utf-8 then latin-1 for CSV.
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Supported formats: CSV, XLSX, XLS")
        if df.empty:
            raise ValueError("The uploaded file is empty")
        return df
    except Exception as e:
        raise Exception(f"Error reading file: {e}")
def coerce_datetime_columns(df: pd.DataFrame, min_parse_ratio: float = 0.8) -> pd.DataFrame:
    """
    Try to convert object columns to datetime if a high ratio of values parse as datetimes.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            non_null = df_copy[col].dropna().astype(str)
            if len(non_null) == 0:
                continue
            # try parsing
            parsed = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
            ratio = parsed.notna().mean()
            if ratio >= min_parse_ratio:
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass
    return df_copy


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    profile["shape"] = df.shape
    profile["dtypes"] = df.dtypes.to_dict()
    profile["missing_by_col"] = df.isnull().sum().sort_values(ascending=False)
    profile["duplicates"] = int(df.duplicated().sum())
    profile["total_missing"] = int(df.isnull().sum().sum())
    profile["numeric_cols"] = df.select_dtypes(include=[np.number]).columns.tolist()
    profile["categorical_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
    profile["datetime_cols"] = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()
    if profile["numeric_cols"]:
        profile["numeric_summary"] = df[profile["numeric_cols"]].describe()
    else:
        profile["numeric_summary"] = pd.DataFrame()
    if profile["categorical_cols"]:
        profile["categorical_cardinality"] = {c: int(df[c].nunique(dropna=True)) for c in profile["categorical_cols"]}
    else:
        profile["categorical_cardinality"] = {}
    return profile


def suggest_cleaning_actions(profile: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    if profile["duplicates"] > 0:
        suggestions.append(f"Remove {profile['duplicates']} duplicate rows")
    if profile["total_missing"] > 0:
        pct = (profile["total_missing"] / (profile["shape"][0] * profile["shape"][1])) * 100
        suggestions.append(f"Handle missing values ({pct:.1f}% of total cells)")
    for col, card in profile.get("categorical_cardinality", {}).items():
        if card > profile["shape"][0] * 0.8:
            suggestions.append(f"High cardinality column '{col}' ({card}) - consider alternatives")
    # detect possible datetime names
    datetime_candidates = [c for c in profile.get("categorical_cols", []) if any(k in c.lower() for k in ["date", "time", "created", "updated"])]
    if datetime_candidates:
        suggestions.append(f"Consider parsing datetime columns: {', '.join(datetime_candidates)}")
    if profile.get("numeric_cols"):
        suggestions.append("Consider capping numeric outliers using IQR")
    return suggestions


def apply_iqr_caps(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return series.clip(lower=low, upper=high)


def auto_clean_data(df: pd.DataFrame, numeric_impute: str = "median", categorical_impute: str = "mode", cap_outliers: bool = False) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = coerce_datetime_columns(df_clean)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    for c in numeric_cols:
        if df_clean[c].isnull().any():
            if numeric_impute == "median":
                fill = df_clean[c].median()
            elif numeric_impute == "mean":
                fill = df_clean[c].mean()
            else:
                fill = 0
            df_clean[c] = df_clean[c].fillna(fill)
    for c in categorical_cols:
        if df_clean[c].isnull().any():
            if categorical_impute == "mode":
                modev = df_clean[c].mode(dropna=True)
                fill = modev.iloc[0] if not modev.empty else "Unknown"
            else:
                fill = "Unknown"
            df_clean[c] = df_clean[c].fillna(fill)
    if cap_outliers:
        for c in numeric_cols:
            df_clean[c] = apply_iqr_caps(df_clean[c])
    return df_clean


def get_correlation_pairs(df: pd.DataFrame, top_k: int = 3) -> List[Tuple[str, str, float]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            val = corr.loc[a, b]
            if not pd.isna(val):
                pairs.append((a, b, val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_k]


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return first datetime-like column with non-null values.
    """
    for col in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].dropna().shape[0] > 0:
                return col
        except Exception:
            continue
    return None


def calculate_month_over_month(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Defensive month-over-month calculation.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    tmp = df[[date_col, value_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(tmp[date_col]):
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return pd.DataFrame()
    tmp["year_month"] = tmp[date_col].dt.to_period("M")
    monthly = tmp.groupby("year_month")[value_col].mean().reset_index()
    monthly["month"] = monthly["year_month"].dt.to_timestamp()
    monthly["mom_pct"] = monthly[value_col].pct_change() * 100
    return monthly[["month", value_col, "mom_pct"]].dropna()


def detect_zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    s = series.dropna()
    if s.empty:
        return pd.DataFrame()
    z = np.abs((s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1))
    an = s[z >= threshold]
    return pd.DataFrame({"index": an.index, "value": an.values, "z_score": z[z >= threshold].values})


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.save()
    return output.getvalue()


# -------------------- Main layout --------------------
st.title("Excel-Like Data App 📊")
st.caption("Upload → Clean → Visualize → Discover insights from your data")

# -------------------- Sidebar: Upload & settings --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"], help="Upload a CSV or Excel file to get started")

    if uploaded_file is not None:
        try:
            with st.spinner("Reading file..."):
                st.session_state.raw_df = read_file(uploaded_file)
                # immediately try to coerce date columns for smoother downstream behavior
                try:
                    st.session_state.raw_df = coerce_datetime_columns(st.session_state.raw_df)
                except Exception:
                    # non-fatal; keep raw dataframe if coercion fails
                    pass
            st.success(f"✅ File loaded: {st.session_state.raw_df.shape[0]} rows, {st.session_state.raw_df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.raw_df = None

    # Cleaning options (only visible after upload)
    if st.session_state.raw_df is not None:
        st.subheader("🧹 Cleaning Options")
        numeric_impute = st.selectbox("Numeric Imputation", ["median", "mean", "zero"], help="Strategy for filling missing numeric values")
        categorical_impute = st.selectbox("Categorical Imputation", ["mode", "Unknown"], help="Strategy for filling missing categorical values")
        cap_outliers = st.checkbox("Cap Numeric Outliers (IQR)", value=False, help="Apply IQR-based outlier capping to numeric columns")

        if st.button("🔧 Auto-Clean", type="primary"):
            try:
                with st.spinner("Cleaning data..."):
                    st.session_state.clean_df = auto_clean_data(st.session_state.raw_df, numeric_impute=numeric_impute, categorical_impute=categorical_impute, cap_outliers=cap_outliers)
                st.success("✅ Data cleaned successfully!")
            except Exception as e:
                st.error(f"❌ Error cleaning data: {str(e)}")

# -------------------- Main content --------------------
if st.session_state.raw_df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Preview & Profile", "🧹 Cleaning", "📈 Visualization", "💡 Auto-Insights"])

    # ----------- Tab 1: Preview & Profile -----------
    with tab1:
        st.header("📊 Data Preview & Profile")
        current_df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df

        st.subheader("Data Preview")
        preview_rows = min(100, len(current_df))
        st.dataframe(current_df.head(preview_rows), use_container_width=True)

        profile = profile_data(current_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{profile['shape'][0]:,}")
        c2.metric("Total Columns", profile["shape"][1])
        c3.metric("Duplicate Rows", f"{profile['duplicates']:,}")
        c4.metric("Missing Cells", f"{profile['total_missing']:,}")

        left, right = st.columns(2)
        with left:
            with st.expander("🔍 Missing Values by Column"):
                if profile["total_missing"] > 0:
                    missing = profile["missing_by_col"][profile["missing_by_col"] > 0]
                    st.dataframe(missing, use_container_width=True)
                else:
                    st.info("No missing values found!")
            with st.expander("📂 Categorical Cardinality"):
                if profile["categorical_cardinality"]:
                    card_df = pd.DataFrame(list(profile["categorical_cardinality"].items()), columns=["Column", "Unique Values"]).sort_values("Unique Values", ascending=False)
                    st.dataframe(card_df, use_container_width=True)
                else:
                    st.info("No categorical columns found!")
        with right:
            with st.expander("📊 Numeric Summary"):
                if not profile["numeric_summary"].empty:
                    st.dataframe(profile["numeric_summary"], use_container_width=True)
                else:
                    st.info("No numeric columns found!")
            with st.expander("💡 Suggested Actions"):
                suggestions = suggest_cleaning_actions(profile)
                if suggestions:
                    for s in suggestions:
                        st.write("•", s)
                else:
                    st.info("No specific actions suggested - data looks clean!")

    # ----------- Tab 2: Cleaning -----------
    with tab2:
        st.header("🧹 Data Cleaning")
        if st.session_state.clean_df is not None:
            st.success("✅ Data has been cleaned using the selected strategies")
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Before Cleaning")
                raw_profile = profile_data(st.session_state.raw_df)
                st.metric("Rows", f"{raw_profile['shape'][0]:,}")
                st.metric("Missing Cells", f"{raw_profile['total_missing']:,}")
                st.metric("Duplicates", f"{raw_profile['duplicates']:,}")
            with col_b:
                st.subheader("After Cleaning")
                clean_profile = profile_data(st.session_state.clean_df)
                st.metric("Rows", f"{clean_profile['shape'][0]:,}")
                st.metric("Missing Cells", f"{clean_profile['total_missing']:,}")
                st.metric("Duplicates", f"{clean_profile['duplicates']:,}")
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.clean_df.head(100), use_container_width=True)
        else:
            st.info("Use the sidebar settings to configure and apply cleaning operations")

        # Download current dataset (CSV & Excel)
        current_df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df
        if current_df is not None:
            csv_buffer = io.StringIO()
            current_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="📥 Download Current Dataset (CSV)",
                data=csv_data,
                file_name=f"dataset_{'cleaned' if st.session_state.clean_df is not None else 'original'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            # Excel download
            try:
                excel_bytes = to_excel_bytes(current_df)
                st.download_button(
                    label="📥 Download Current Dataset (Excel .xlsx)",
                    data=excel_bytes,
                    file_name=f"dataset_{'cleaned' if st.session_state.clean_df is not None else 'original'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                # don't crash if excel writer fails
                pass

    # ----------- Tab 3: Visualization -----------
    with tab3:
        st.header("📈 Data Visualization")
        current_df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df

        if current_df is None or len(current_df) == 0:
            st.warning("No data available for visualization")
        else:
            col_left, col_right = st.columns([1, 2])
            with col_left:
                chart_type = st.selectbox("Chart Type", ["histogram", "bar", "line", "scatter", "box"], help="Select chart type")
                all_columns = current_df.columns.tolist()
                numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()

                x_col = st.selectbox("X Column", all_columns, help="Select X-axis column")
                y_col = None
                if chart_type in ["scatter", "line", "bar"]:
                    if numeric_columns:
                        y_col = st.selectbox("Y Column (numeric, optional)", [None] + numeric_columns, help="Select Y-axis column")
                    else:
                        st.warning("No numeric columns available for Y-axis")

                color_col = st.selectbox("Color/Group Column (Optional)", [None] + all_columns, help="Optional group/color column")
                groupby_cols = st.multiselect("Group by (optional, for aggregation)", options=all_columns, help="Select columns to group by (makes pivot-like aggregations)")

                aggregation = "auto"
                if chart_type in ["bar", "line"]:
                    aggregation = st.selectbox("Aggregation Method", ["auto", "count", "sum", "mean", "median"], help="Aggregation for bar/line")

                if st.button("Generate Chart"):
                    try:
                        fig = None

                        # Helper: determine agg func
                        def resolve_agg(func_name: str, x_is_numeric: bool):
                            if func_name == "auto":
                                return "mean" if x_is_numeric else "count"
                            return func_name

                        if chart_type == "histogram":
                            if x_col in numeric_columns:
                                fig = px.histogram(current_df, x=x_col, color=color_col, title=f"Histogram of {x_col}")
                            else:
                                st.warning("Histogram requires numeric X column")
                        elif chart_type == "bar":
                            agg_func = resolve_agg(aggregation, x_col in numeric_columns)
                            if groupby_cols and y_col:
                                plot_df = current_df.groupby(groupby_cols)[y_col].agg(agg_func).reset_index()
                                # if multiple groupby cols, use the first as x for simple bar display
                                if len(groupby_cols) == 1:
                                    fig = px.bar(plot_df, x=groupby_cols[0], y=y_col, color=color_col, title=f"{y_col} by {groupby_cols[0]}")
                                else:
                                    # show grouped bar using treemap-like fallback (or show table)
                                    fig = px.bar(plot_df, x=groupby_cols[0], y=y_col, color=groupby_cols[1] if len(groupby_cols) > 1 else color_col, title=f"Aggregated {y_col}")
                            else:
                                if agg_func == "count":
                                    plot_df = current_df[x_col].value_counts().reset_index()
                                    plot_df.columns = [x_col, "count"]
                                    fig = px.bar(plot_df, x=x_col, y="count", title=f"Count by {x_col}")
                                elif x_col in numeric_columns and (y_col is None):
                                    fig = px.histogram(current_df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
                                elif y_col:
                                    # aggregate by x_col
                                    agg = resolve_agg(aggregation, True)
                                    plot_df = current_df.groupby(x_col)[y_col].agg(agg).reset_index()
                                    fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, title=f"{agg} of {y_col} by {x_col}")
                                else:
                                    st.warning("Select appropriate columns for bar chart")
                        elif chart_type == "line":
                            if y_col is None:
                                st.warning("Line chart needs a Y column")
                            else:
                                if pd.api.types.is_datetime64_any_dtype(current_df[x_col]) or any("date" in str(x_col).lower() for x_col in [x_col]):
                                    df_plot = current_df[[x_col, y_col]].copy()
                                    df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="coerce")
                                    df_plot = df_plot.dropna(subset=[x_col, y_col]).set_index(x_col).resample("D")[y_col].mean().reset_index()
                                    fig = px.line(df_plot, x=x_col, y=y_col, title=f"{y_col} over time")
                                else:
                                    if groupby_cols:
                                        agg_func = resolve_agg(aggregation, True)
                                        plot_df = current_df.groupby(groupby_cols)[y_col].agg(agg_func).reset_index()
                                        fig = px.line(plot_df, x=groupby_cols[0], y=y_col, color=color_col, title=f"{y_col} by {groupby_cols[0]}")
                                    else:
                                        fig = px.line(current_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
                        elif chart_type == "scatter":
                            if y_col is None:
                                st.warning("Scatter requires X and Y numeric columns")
                            else:
                                fig = px.scatter(current_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                        elif chart_type == "box":
                            if y_col:
                                fig = px.box(current_df, x=color_col if color_col else None, y=y_col, title=f"Box plot of {y_col}")
                            else:
                                if x_col in numeric_columns:
                                    fig = px.box(current_df, y=x_col, color=color_col, title=f"Box plot of {x_col}")
                                else:
                                    st.warning("Box plot needs a numeric column")

                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to generate chart: {e}")

    # ----------- Tab 4: Auto-Insights -----------
    with tab4:
        st.header("💡 Auto-Generated Insights")
        current_df = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df

        if current_df is None:
            st.info("Upload a dataset first to generate insights.")
        else:
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_metric = None
            if numeric_cols:
                selected_metric = st.selectbox("Select numeric metric for time-trends / anomalies", numeric_cols, index=0)
            else:
                st.info("No numeric columns available for time trends or anomaly detection.")

            if st.button("🔍 Generate Insights", type="primary"):
                with st.spinner("Analyzing data..."):
                    insights: List[str] = []
                    profile = profile_data(current_df)
                    insights.append(f"**Dataset Overview**: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns with {profile['total_missing']:,} missing cells ({(profile['total_missing']/(profile['shape'][0]*profile['shape'][1])*100):.1f}% of total)")

                    # Top correlations
                    if len(profile["numeric_cols"]) >= 2:
                        corr_pairs = get_correlation_pairs(current_df, top_k=3)
                        if corr_pairs:
                            insights.append("**Top Correlations**:")
                            for a, b, val in corr_pairs:
                                insights.append(f"  • {a} ↔ {b}: r = {val:.3f}")

                    # Segment analysis (categorical)
                    categorical_cols = [col for col, card in profile.get("categorical_cardinality", {}).items() if 2 <= card <= 30]
                    if categorical_cols and profile["numeric_cols"]:
                        cat = categorical_cols[0]
                        num = profile["numeric_cols"][0]
                        seg_stats = current_df.groupby(cat)[num].mean().sort_values()
                        if len(seg_stats) > 1:
                            top = seg_stats.index[-1]
                            bot = seg_stats.index[0]
                            insights.append(f"**Segment Analysis**: '{top}' has highest average {num} ({seg_stats.iloc[-1]:.2f}), '{bot}' has lowest ({seg_stats.iloc[0]:.2f})")

                    # Time trend
                    time_col = detect_time_column(current_df)
                    if time_col and selected_metric:
                        mom = calculate_month_over_month(current_df, time_col, selected_metric)
                        if not mom.empty and "mom_pct" in mom.columns:
                            latest = mom.iloc[-1]
                            insights.append(f"**Time Trend**: Latest month-over-month change for {selected_metric} is {latest['mom_pct']:+.1f}% in {latest['month'].strftime('%b %Y')}")
                            # show small plot
                            with st.expander("📈 Monthly Trend (expand)"):
                                fig = px.line(mom, x="month", y=selected_metric, markers=True, title=f"Monthly mean of {selected_metric}")
                                st.plotly_chart(fig, use_container_width=True)

                    # Anomalies
                    if selected_metric:
                        anom = detect_zscore_anomalies(current_df[selected_metric])
                        if not anom.empty:
                            insights.append(f"**Anomalies Detected**: {len(anom)} outliers in '{selected_metric}' using z-score ≥ 3.0")
                        else:
                            insights.append(f"**Data Quality**: No significant outliers detected in '{selected_metric}' (z-score < 3.0)")

                    # Display insights
                    st.subheader("Key Findings")
                    for line in insights:
                        if line.startswith("**") and ":" in line:
                            p0, p1 = line.split(":", 1)
                            st.markdown(f"{p0}:** {p1}")
                        else:
                            st.markdown(line)

                    # Download insights as markdown
                    insights_text = "\n".join([l.replace("**", "").replace("  • ", "- ") for l in insights])
                    md = f"# Data Analysis Insights\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{insights_text}\n\n---\nGenerated by Clean • Viz • Insights\n"
                    st.download_button(label="📥 Download Insights (Markdown)", data=md, file_name=f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
else:
    # Welcome / instructions
    st.markdown(
        """
    ## 🚀 Getting Started
    
    **Upload your data** using the file uploader in the sidebar to begin your analysis journey.
    
    ### What you can do:
    1. **📊 Preview & Profile**: Explore structure, issues, cleaning recommendations
    2. **🧹 Cleaning**: Auto clean (duplicates, NAs) and export
    3. **📈 Visualization**: Interactive charts (bar, line, histogram, scatter, box)
    4. **💡 Auto-Insights**: Correlations, segment highlights, time-trends, anomalies
    
    Supported formats: CSV, XLSX
    """
    )

import streamlit as st

# === Sidebar Branding ===
st.sidebar.title("📊 Data Analytics App")
st.sidebar.markdown("---")
st.sidebar.markdown("👤 **Developed by Usama Butt**")
st.sidebar.markdown("---")

# === Main Content ===
st.title("My Data Analytics Dashboard")
st.write("Data is loading / displaying here...")


