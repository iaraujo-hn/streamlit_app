"""
QA Tool – A Streamlit application for comparing two CSV/Excel files

Features:
- File upload with automatic encoding detection
- Automatic cleaning and conversion of ID and date columns
- Automatic and manual matching of column names using Levenshtein distance
- Group-by and metric selection for side-by-side comparison
- Highlighting of significant discrepancies with a download option for results
"""

import streamlit as st
import pandas as pd
import re
import chardet
import Levenshtein

# ---------------------------------- Configurations ---------------------------------- #

# Columns to be included/excluded in the group-by selection
default_groupby_columns = ['placement_id', 'campaign_id', "creative_id", 'site_id']
excluded_groupby_columns = [
    'impressions', 'clicks', 'Click', 'spend', 'sessions', 'page_views', 'revenue', 'conversions'
]

# Define common column mappings for automatic matching (Main File column: File 2 column)
column_mapping = {
    "site_dcm": "Site (CM360)",
    'mmm_roas': 'roas',
    "Cost": "spend",
}

# ---------------------------------- Caching and File Reading ---------------------------------- #

@st.cache_data(show_spinner=True)
def read_file(file) -> pd.DataFrame:
    """
    Read CSV or Excel file with automatic encoding detection and user-friendly error handling.
    Also cleans ID columns to avoid issues with trailing decimals.
    """
    with st.spinner('Loading file...'):
        df = None
        error_message = None

        try:
            if file.name.endswith('.csv'):
                raw_bytes = file.read(10000)  # Read a portion for encoding detection
                detected_encoding = chardet.detect(raw_bytes)['encoding']
                file.seek(0)  # Reset pointer
                df = pd.read_csv(file, encoding=detected_encoding if detected_encoding else 'utf-8')
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl')
            else:
                st.error("❌ Unsupported file type. Please upload a **CSV or Excel file**.")
                return None

            # Validate the loaded dataframe
            if df.empty or df.columns.notna().sum() <= 1:
                error_message = "⚠️ The file appears empty or incorrectly formatted. Please verify its contents."
        except UnicodeDecodeError:
            error_message = ("⚠️ There was an issue reading the file due to invalid characters. "
                             "Try saving it with UTF-8 encoding.")
        except pd.errors.EmptyDataError:
            error_message = "⚠️ The file appears to be empty. Please verify its contents."
        except pd.errors.ParserError:
            error_message = "⚠️ The file format may be corrupted or incorrect. Please try re-saving the file."
        except Exception as e:
            error_message = f"⚠️ Unexpected error: {str(e)}"

        if error_message:
            st.error(error_message)
            st.markdown(
                """
                **Troubleshooting Tips:**
                - Verify that the file is not empty.
                - Ensure the file is in CSV or Excel format.
                - For CSV files with character issues, re-save using UTF-8 encoding.
                - For Excel files, try saving as a new file.
                """
            )
            return None

        df = clean_id_columns(df)
        return df

def clean_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns containing 'id' to strings after converting numeric values to integers.
    This fixes mismatches caused by trailing decimals in numeric IDs.
    """
    for col in df.columns:
        if "id" in col.lower():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0).apply(lambda x: str(int(x)) if pd.notnull(x) else "0")
            else:
                df[col] = df[col].astype(str)
    return df

# ---------------------------------- Data Cleaning and Conversion ---------------------------------- #

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns likely to be dates into datetime format using regex detection.
    """
    date_keywords = ['date', 'month', 'year']
    date_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}')

    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            sample_values = df[col].dropna().astype(str).head(10)
            if sample_values.apply(lambda x: bool(date_pattern.search(x))).any():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
    return df

def clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Remove non-numeric characters (except periods) and convert a Series to numeric.
    """
    cleaned_series = (
        series.astype(str)
              .str.replace(r"[^0-9.]", "", regex=True)
              .str.replace(r"\.+", ".", regex=True)
    )
    return pd.to_numeric(cleaned_series, errors="coerce")

def ensure_numeric_and_clean(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """
    Ensure specified columns are numeric by cleaning and converting them.
    Skips columns that are used for grouping.
    """
    for col in numeric_columns:
        if col in df.columns and col not in default_groupby_columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    return df

# ---------------------------------- Column Matching ---------------------------------- #

def find_best_match(input_cols: list, columns_list: list) -> dict:
    """
    For each column in input_cols, find the best matching column from columns_list using Levenshtein distance.
    Matches are only accepted if within a threshold (3 by default) and non-conflicting.
    """
    input_cols_lower = [col.lower() for col in input_cols]
    columns_list_lower = [col.lower() for col in columns_list]
    used_matches = set()
    best_matches = {}

    for orig_col, input_col_lower in zip(input_cols, input_cols_lower):
        best_match = None
        best_distance = float('inf')

        # First, check for an exact match
        for i, col in enumerate(columns_list_lower):
            if col in used_matches:
                continue
            if col == input_col_lower:
                best_match = columns_list[i]
                best_distance = 0
                break

        # If no exact match, apply Levenshtein distance
        if not best_match:
            for i, col in enumerate(columns_list_lower):
                if col in used_matches:
                    continue
                distance = Levenshtein.distance(input_col_lower, col)
                # Additional protection to avoid mismatches between columns with "id" differences
                is_conflicting = (
                    (input_col_lower in col or col in input_col_lower)
                    and (("id" in input_col_lower and "id" not in col) or ("id" not in input_col_lower and "id" in col))
                )
                if distance < best_distance and not is_conflicting:
                    best_match = columns_list[i]
                    best_distance = distance

        # Accept the match only if within threshold; otherwise, keep the original name.
        if best_match and best_distance <= 3:
            best_matches[orig_col] = best_match
            used_matches.add(best_match.lower())
        else:
            best_matches[orig_col] = orig_col

    return best_matches

def auto_match_columns(df1: pd.DataFrame, df2: pd.DataFrame, mapping: dict):
    """
    Automatically match and rename columns between two DataFrames.
    Priority is given to the provided mapping, then to auto-matching based on Levenshtein distance.
    """
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)
    column_replacements = {}

    # Apply predefined mapping if both columns exist
    for col1, col2 in mapping.items():
        if col1 in columns_1 and col2 in columns_2:
            column_replacements[col2] = col1
        elif col2 in columns_1 and col1 in columns_2:
            column_replacements[col1] = col2

    # Rename columns in both dataframes based on predefined mapping
    df1.rename(columns=column_replacements, inplace=True)
    df2.rename(columns=column_replacements, inplace=True)

    # Auto-match remaining columns from File 2 to File 1
    auto_matches = find_best_match(columns_2, columns_1)
    for col, matched_col in auto_matches.items():
        if matched_col != col and col not in column_replacements:
            column_replacements[col] = matched_col

    df2.rename(columns=column_replacements, inplace=True)
    return df1, df2, column_replacements

# ---------------------------------- UI Display Functions ---------------------------------- #

def display_uploaded_data(file_1, file_2):
    """
    Read, clean, and display previews of the uploaded files.
    """
    df1 = read_file(file_1)
    df2 = read_file(file_2)

    if df1 is not None and df2 is not None:
        df1 = convert_date_columns(df1)
        df2 = convert_date_columns(df2)

        # Use session state for toggling the data preview
        if "show_data" not in st.session_state:
            st.session_state.show_data = False

        if st.button("Toggle Data Preview"):
            st.session_state.show_data = not st.session_state.show_data

        if st.session_state.show_data:
            tab1, tab2 = st.tabs(["Main File", "File 2"])
            with tab1:
                st.write("#### Main File Preview (first 10 rows)")
                st.dataframe(df1.head(10))
            with tab2:
                st.write("#### File 2 Preview (first 10 rows)")
                st.dataframe(df2.head(10))
        return df1, df2
    return None, None

def manual_edit_columns(df1: pd.DataFrame, df2: pd.DataFrame, column_replacements_auto: dict):
    """
    Allow manual editing of File 2's column names to match File 1.
    Display the columns side-by-side with color highlights for already matching columns.
    """
    # Identify matching and non-matching columns
    matching_columns = [col for col in df1.columns if col in df2.columns]
    non_matching_columns_1 = [col for col in df1.columns if col not in df2.columns]
    non_matching_columns_2 = [col for col in df2.columns if col not in df1.columns]

    ordered_columns_1 = matching_columns + non_matching_columns_1
    ordered_columns_2 = matching_columns + non_matching_columns_2

    max_len = max(len(ordered_columns_1), len(ordered_columns_2))
    columns_df = pd.DataFrame({
        "Main File": ordered_columns_1 + [""] * (max_len - len(ordered_columns_1)),
        "File 2": ordered_columns_2 + [""] * (max_len - len(ordered_columns_2))
    })

    # CSS styling for table
    st.markdown(
        """
        <style>
        div[data-testid="stTable"] {
            width: 100% !important;
        }
        table {
            width: 100% !important;
            table-layout: auto !important;
        }
        thead th {
            text-align: left !important;
            font-weight: bold;
            background-color: #f0f2f6;
        }
        tbody td {
            padding: 12px;
            word-wrap: break-word;
            max-width: 600px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def highlight_matching(val):
        return "background-color: #c6f5c6; font-weight: bold;" if val in matching_columns else ""

    if st.checkbox("Show Column Mapping Table"):
        st.write("**Green-highlighted columns already match between the files.**")
        if column_replacements_auto:
            tooltip_text = "<br>".join([f"{old} → {new}" for old, new in column_replacements_auto.items()])
            st.markdown(
                f'<div title="{tooltip_text}" style="cursor: help;">ℹ️ Automatically Renamed Columns (hover to view)</div>',
                unsafe_allow_html=True
            )
        st.table(columns_df.style.applymap(highlight_matching))

    # Allow manual renaming of selected File 2 columns
    manual_replacements = {}
    st.write("##### Select columns from File 2 to rename:")
    cols_to_rename = st.multiselect("Choose File 2 columns:", list(df2.columns))
    if cols_to_rename:
        st.write("##### Rename the selected columns to match File 1:")
        for col in cols_to_rename:
            new_name = st.selectbox(
                f"Rename '{col}' to:",
                [""] + list(df1.columns),
                key=f"rename_{col}"
            )
            if new_name and new_name != col:
                manual_replacements[col] = new_name
                df2.rename(columns={col: new_name}, inplace=True)
        if manual_replacements:
            st.success("Manual renaming applied:")
            for old, new in manual_replacements.items():
                st.write(f"**{old}** → **{new}**")
    return df1, df2, manual_replacements

def prepare_group_and_metric_options(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Prepare options for group-by and metric selection based on common columns.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))
    groupby_options = sorted([col for col in common_columns if col not in excluded_groupby_columns])
    valid_default_groupby = [col for col in default_groupby_columns if col in groupby_options and "id" not in col.lower()]
    
    # Convert default groupby columns to string type
    for col in default_groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)
    
    # Identify numeric columns for metrics (exclude the valid default groupby columns)
    numeric_columns = [col for col in common_columns 
                       if pd.api.types.is_numeric_dtype(df1[col]) and col not in valid_default_groupby]
    
    return groupby_options, valid_default_groupby, numeric_columns

def group_and_compare(df1: pd.DataFrame, df2: pd.DataFrame, groupby_columns: list, selected_metrics: list):
    """
    Group the two dataframes by the specified columns and compare selected numeric metrics.
    Also highlights rows where discrepancies exceed a defined threshold.
    """
    # Ensure grouping columns are treated as strings
    for col in default_groupby_columns + groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    # Clean and convert metric columns to numeric
    df1 = ensure_numeric_and_clean(df1, [m for m in selected_metrics if m not in default_groupby_columns])
    df2 = ensure_numeric_and_clean(df2, [m for m in selected_metrics if m not in default_groupby_columns])

    # Rename metric columns to distinguish their source
    df1 = df1.rename(columns={col: f"{col} - Main File" for col in selected_metrics})
    df2 = df2.rename(columns={col: f"{col} - File 2" for col in selected_metrics})

    # Group by the chosen columns
    df1_grouped = df1.groupby(groupby_columns)[[f"{col} - Main File" for col in selected_metrics]].sum().reset_index()
    df2_grouped = df2.groupby(groupby_columns)[[f"{col} - File 2" for col in selected_metrics]].sum().reset_index()
    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns, how="outer").fillna(0)

    results = merged_df[groupby_columns].copy()
    discrepancy_mask = pd.Series([False] * len(merged_df))

    # Compare metrics and calculate differences and percentage differences
    for col in selected_metrics:
        col_A = f"{col} - Main File"
        col_B = f"{col} - File 2"
        diff_col = f"{col} Difference"
        pct_diff_col = f"{col} % Difference"

        if col_A in merged_df.columns and col_B in merged_df.columns:
            merged_df[diff_col] = merged_df[col_A] - merged_df[col_B]

            def calc_pct_diff(a, b):
                if a == 0 and b == 0:
                    return 0
                elif b == 0:
                    return float('inf')
                else:
                    return (a - b) / abs(b) * 100

            merged_df[pct_diff_col] = merged_df.apply(lambda row: calc_pct_diff(row[col_A], row[col_B]), axis=1)
            results[col_A] = merged_df[col_A]
            results[col_B] = merged_df[col_B]
            results[diff_col] = merged_df[diff_col]
            results[pct_diff_col] = merged_df[pct_diff_col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else "∞%")
            discrepancy_mask |= merged_df[pct_diff_col].abs() > 0.5

    if not results.empty:
        results["Sig. Discrepancy"] = discrepancy_mask.map({True: "Yes", False: "No"})

    # Store results in session state for later use (e.g. downloading)
    st.session_state["comparison_results"] = results
    st.session_state["discrepancy_mask"] = discrepancy_mask
    st.session_state["groupby_columns"] = groupby_columns

    st.write("### Side-by-Side Comparison")
    st.dataframe(results)

    # Display discrepancy summary
    if discrepancy_mask.any():
        num_discrepancies = discrepancy_mask.sum()
        total_rows = len(results)
        st.error(f"Found {num_discrepancies} rows with discrepancies larger than 0.5% ({(num_discrepancies/total_rows)*100:.2f}% of the data).")
    else:
        st.success("✅ No discrepancies greater than 0.5% found.")

def display_discrepancies():
    """
    Offer a toggle to view rows with significant discrepancies.
    """
    if "comparison_results" not in st.session_state or "discrepancy_mask" not in st.session_state:
        return

    results = st.session_state["comparison_results"]
    discrepancy_mask = st.session_state["discrepancy_mask"]

    if discrepancy_mask.any():
        if st.checkbox("Show Discrepancy Table"):
            st.write("### Significant Discrepancies (Rows with >0.5% difference)")
            st.dataframe(results[discrepancy_mask])

def download_results():
    """
    Provide a download button to export the comparison results as a CSV.
    """
    if "comparison_results" in st.session_state:
        csv = st.session_state["comparison_results"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Comparison Results as CSV",
            data=csv,
            file_name='comparison_results.csv',
            mime='text/csv'
        )

# ---------------------------------- Main App Layout ---------------------------------- #

def main():
    st.set_page_config(page_title="QA Tool", layout="wide")
    st.title("QA Tool")
    st.write("This tool compares data between two files, highlighting discrepancies in selected metrics.")

    # Sidebar instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. **Upload Files:** Upload your Main File and the File to compare (CSV or Excel).
        2. **Preview Data:** Toggle the data preview to inspect the first 10 rows.
        3. **Auto-Match Columns:** The tool will automatically match similar columns.
        4. **Manual Edit:** (Optional) Manually adjust column names for better matching.
        5. **Group & Compare:** Select group-by columns and metrics, then click **Compare**.
        6. **Review & Download:** Examine discrepancies and download the results if needed.
        """
    )

    # File upload
    file_1 = st.file_uploader("**Choose the first file (Main File):**", type=["csv", "xlsx"], key="file_1")
    file_2 = st.file_uploader("**Choose the second file (Comparison File):**", type=["csv", "xlsx"], key="file_2")

    if file_1 and file_2:
        df1, df2 = display_uploaded_data(file_1, file_2)

        st.markdown("---")
        if df1 is not None and df2 is not None:
            # Auto-match columns using predefined mapping and fuzzy matching
            df1, df2, auto_replacements = auto_match_columns(df1, df2, column_mapping)

            total_cols = len(df1.columns)
            matching_cols = sum(1 for col in df2.columns if col in df1.columns)
            st.write("#### Column Matching")
            if matching_cols == total_cols:
                st.success(f"All {total_cols} columns in File 2 match File 1!")
            elif matching_cols > 0:
                st.info(f"{matching_cols} out of {total_cols} columns in File 2 match File 1.")
            else:
                st.warning("None of the columns in File 2 match File 1. Consider using the manual edit option below.")

            # Toggle manual column editing
            if "show_manual_edit" not in st.session_state:
                st.session_state.show_manual_edit = False
            if st.button("Toggle Manual Column Editing"):
                st.session_state.show_manual_edit = not st.session_state.show_manual_edit
            if st.session_state.show_manual_edit:
                df1, df2, manual_replacements = manual_edit_columns(df1, df2, auto_replacements)

            st.markdown("---")
            # Prepare group-by and metric options
            groupby_options, valid_default_groupby, numeric_columns = prepare_group_and_metric_options(df1, df2)
            groupby_columns = st.multiselect("Select columns to group by:", groupby_options, default=valid_default_groupby)
            selected_metrics = st.multiselect("Select numeric metrics to compare:", numeric_columns)

            if st.button("Compare"):
                if groupby_columns and selected_metrics:
                    group_and_compare(df1, df2, groupby_columns, selected_metrics)
                    display_discrepancies()
                    download_results()
                else:
                    st.warning("Please select at least one group-by column and one metric to compare.")

if __name__ == '__main__':
    main()