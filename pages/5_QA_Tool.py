# ---------------------------------- Package Errors Fix - Run in your terminal ---------------------------------- #
# pip show streamlit pandas numpy openpyxl spacy rapidfuzz Levenshtein
# python -m spacy download en_core_web_sm

# ---------------------------------- Import Libraries ---------------------------------- #
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import spacy
import re  # regex
import Levenshtein

# Load spacy model for NLP matching
# nlp = spacy.load('en_core_web_sm')
# try:
#     nlp = spacy.load('en_core_web_sm')
# except OSError:
#     st.error("Spacy model 'en_core_web_sm' not found. .")

# Columns to exclude from the group by selection. This will help to maintain the group by selection cleaner for the final user
excluded_groupby_columns = [
    'impressions', 'clicks', 'Click', 'spend', 'sessions', 'page_views', 'revenue', 'conversions'
]

# ---------------------------------- File Upload ---------------------------------- #

@st.cache_data
def read_file(file):
    """Read csv or excel file based on file extension"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

# ---------------------------------- Data Processing ---------------------------------- #
# Testing Levenshtein distance as the matching process. We still need to test it more to see if there are any flaws
def find_best_match(input_cols, columns_list):
    """Find the best match for each input column using levenshtein distance and avoid duplications."""
    input_cols_lower = [col.lower() for col in input_cols]
    columns_list_lower = [col.lower() for col in columns_list]
    used_matches = set()
    best_matches = {}

    for input_col, input_col_lower in zip(input_cols, input_cols_lower):
        best_match = None
        best_distance = float('inf')

        for i, col in enumerate(columns_list_lower):
            if col in used_matches:
                continue

            distance = Levenshtein.distance(input_col_lower, col)
            # Ensure the columns share common keywords
            common_words = set(input_col_lower.split()) & set(col.split())
            if distance < best_distance and (common_words or input_col_lower in col or col in input_col_lower):
                best_match = columns_list[i]
                best_distance = distance

        # Only use the best match if within a reasonable threshold and not already matched
        if best_match and best_distance <= 3:
            best_matches[input_col] = best_match
            used_matches.add(best_match.lower())
        else:
            best_matches[input_col] = input_col  # No suitable match found; keep the original name

    return best_matches

# def find_best_match(input_col, columns_list):
# Original matching process using NLP and rapidfuzz
#     """Find the best match for column names using fuzzy matching and NLP"""
#     input_col_lower = input_col.lower()
#     columns_list_lower = [col.lower() for col in columns_list]

#     # Fuzzy matching
#     result = process.extract(input_col_lower, columns_list_lower, scorer=fuzz.ratio, limit=5)
#     best_match = None
#     best_score = 0

#     for match, score, index in result:
#         if score > best_score:
#             best_match = columns_list[index]
#             best_score = score

#     if best_score >= 75:
#         return best_match

#     # NLP similarity
#     input_doc = nlp(input_col_lower)
#     best_similarity = 0

#     for col in columns_list:
#         col_doc = nlp(col.lower())
#         similarity = input_doc.similarity(col_doc)
#         if similarity > best_similarity:
#             best_match = col
#             best_similarity = similarity

#     # It will only return the fuzzy matched words if they are synonyms
#     return best_match if best_similarity > 0.8 else input_col 


def convert_date_columns(df):
    """Convert date related columns to datetime format"""
    date_keywords = ['date', 'week', 'month', 'year']

    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            sample_values = df[col].dropna().head(10).astype(str)
            date_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}') # Regex for date format

            if sample_values.apply(lambda x: bool(date_pattern.search(x))).any():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
    return df


def check_and_convert_to_numeric(df, column_name):
    """Convert a column to numeric and display warnings for problematic values"""
    try:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        problematic_values = df[column_name].isna()
        if problematic_values.any():
            st.warning(f"Column '{column_name}' contains non-numeric values that couldn't be converted:")
            st.write(df.loc[problematic_values, column_name])
        return df[column_name]
    except Exception as e:
        st.error(f"Error converting column '{column_name}' to numeric: {e}")
        return df[column_name]


# ---------------------------------- UI Components ---------------------------------- #

def display_uploaded_data(file_1, file_2):
    """Upload and display data from two files."""
    df1 = read_file(file_1)
    df2 = read_file(file_2)

    if df1 is not None and df2 is not None:
        st.success("Files uploaded successfully")
        df1 = convert_date_columns(df1)
        df2 = convert_date_columns(df2)

        if st.button("View Data"):
            st.write("### Data from File 1")
            st.write(df1.head())
            st.write("### Data from File 2")
            st.write(df2.head())
        return df1, df2
    return None, None

def auto_match_columns(df1, df2):
    """Automatically match and rename columns between two dataframes."""
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)

    auto_match = st.toggle("Enable automatic column renaming", value=True)
    column_replacements = {}

    if auto_match:
        matches = find_best_match(columns_2, columns_1)
        for col, matched_col in matches.items():
            if matched_col != col:
                column_replacements[col] = matched_col
        df2.rename(columns=column_replacements, inplace=True)
    else:
        st.info("Automatic column renaming is disabled. Column names will remain as uploaded.")

    return df1, df2, column_replacements

def manual_edit_columns(df1, df2):
    """Allow manual editing of column names for both dataframe."""
    st.write("### Manual Column Editing")

    st.write("#### Edit Columns in File 1")
    cols1 = st.columns(3)
    for idx, col in enumerate(df1.columns):
        with cols1[idx % 3]: # should the number of columns side by side
            new_col = st.text_input(f"Rename '{col}'", col, key=f"file1_{col}")
            df1.rename(columns={col: new_col}, inplace=True)

    st.write("#### Edit Columns in File 2")
    cols2 = st.columns(3)
    for idx, col in enumerate(df2.columns):
        with cols2[idx % 3]:
            new_col = st.text_input(f"Rename '{col}'", col, key=f"file2_{col}")
            df2.rename(columns={col: new_col}, inplace=True)

    return df1, df2

def group_and_compare(df1, df2, groupby_columns, selected_metrics):
    """Group dataframe and compare metrics."""
    # Identify datetime columns in the DataFrame
    datetime_columns_1 = df1.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    datetime_columns_2 = df2.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

    # Ensure datetime columns are included in the groupby_columns list
    for col in datetime_columns_1:
        if col not in groupby_columns:
            groupby_columns.append(col)

    for col in datetime_columns_2:
        if col not in groupby_columns:
            groupby_columns.append(col)

    # Rename columns to distinguish between files
    df1.columns = [f"{col} - File 1" if col not in groupby_columns else col for col in df1.columns]
    df2.columns = [f"{col} - File 2" if col not in groupby_columns else col for col in df2.columns]

    # Ensure selected metrics are present in the DataFrame
    missing_metrics_1 = [col for col in selected_metrics if f"{col} - File 1" not in df1.columns]
    missing_metrics_2 = [col for col in selected_metrics if f"{col} - File 2" not in df2.columns]

    if missing_metrics_1:
        st.error(f"Selected metrics not found in File 1: {missing_metrics_1}")
        return
    if missing_metrics_2:
        st.error(f"Selected metrics not found in File 2: {missing_metrics_2}")
        return

    # Group by the specified columns and sum the selected metrics
    df1_grouped = df1.groupby(groupby_columns)[[f"{col} - File 1" for col in selected_metrics]].sum().reset_index()
    df2_grouped = df2.groupby(groupby_columns)[[f"{col} - File 2" for col in selected_metrics]].sum().reset_index()

    # Merge the grouped dataframes
    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns)

    results = merged_df[groupby_columns].copy()
    discrepancies_found = False
    discrepancies_list = []

    for col in selected_metrics:
        col_A = f"{col} - File 1"
        col_B = f"{col} - File 2"
        diff_col = f"{col} Difference"
        pct_diff_col = f"{col} % Difference"

        if col_A in merged_df.columns and col_B in merged_df.columns:
            merged_df[col_A] = check_and_convert_to_numeric(merged_df, col_A)
            merged_df[col_B] = check_and_convert_to_numeric(merged_df, col_B)

            results[col_A] = merged_df[col_A]
            results[col_B] = merged_df[col_B]

            # Calculate the difference and percentage difference
            merged_df[diff_col] = merged_df[col_A] - merged_df[col_B]
            merged_df[pct_diff_col] = (merged_df[diff_col] / merged_df[col_B]) * 100

            # Format the percentage difference
            results[diff_col] = merged_df[diff_col]
            results[pct_diff_col] = merged_df[pct_diff_col].apply(lambda x: f"{x:.2f}%")

            # Identify rows with discrepancies greater than 0.5%
            discrepancy_mask = merged_df[pct_diff_col].abs() > 0.5
            discrepancies_list.append(merged_df[discrepancy_mask])

            if discrepancy_mask.any():
                discrepancies_found = True

    st.write("#### Side by Side Comparison")
    st.write(results)

    if discrepancies_found:
        # Concatenate all discrepancies
        significant_discrepancies = pd.concat(discrepancies_list).drop_duplicates()

        # Calculate discrepancy percentage based on grouped rows
        grouped_total_rows = len(merged_df)
        num_discrepancies = len(significant_discrepancies.drop_duplicates(subset=groupby_columns))
        discrepancy_percentage = (num_discrepancies / grouped_total_rows) * 100

        st.error(f"Found {num_discrepancies} rows with discrepancies, representing {discrepancy_percentage:.2f}% of the total grouped data.")
        st.write("#### Significant Discrepancies (Difference > 0.5%)")
        st.write(significant_discrepancies)
    else:
        st.success("No discrepancies greater than 0.5% found.")

def check_and_convert_to_numeric(df, col):
    """Convert column to numeric, coercing errors."""
    return pd.to_numeric(df[col], errors='coerce')
    
# ---------------------------------- Main App ---------------------------------- #

st.title("QA Tool")
st.write("The QA Tool compares data between two files, highlighting discrepancies.")

file_1 = st.file_uploader("Choose the first file", type=["csv", "xlsx"], key="file_1")
file_2 = st.file_uploader("Choose the second file", type=["csv", "xlsx"], key="file_2")

if file_1 and file_2:
    df1, df2 = display_uploaded_data(file_1, file_2)
    if df1 is not None and df2 is not None:
        df1, df2, column_replacements = auto_match_columns(df1, df2)

        # Display renamed columns
        if column_replacements:
            st.write("### Renamed Columns in File 2")
            for old_col, new_col in column_replacements.items():
                st.write(f"**'{old_col}'** was renamed to **'{new_col}'**")

        # Toggle for showing/hiding manual edit options
        if "show_manual_edit" not in st.session_state:
            st.session_state.show_manual_edit = False
            # st.markdown("---")

        if st.button("Manually Edit Column Names"):
            st.session_state.show_manual_edit = not st.session_state.show_manual_edit
            # st.markdown("---")

        # Conditionally display the manual edit options
        if st.session_state.show_manual_edit:
            df1, df2 = manual_edit_columns(df1, df2)

        st.markdown("---")
    
        common_columns = list(set(df1.columns) & set(df2.columns))
        if common_columns:
            st.write("### Select dimensions and metrics:")
            groupby_options = sorted([col for col in common_columns if col not in excluded_groupby_columns])
            groupby_columns = st.multiselect("Select columns to group by:", groupby_options)
            numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(df1[col])]
            selected_metrics = st.multiselect("Select metrics to compare:", numeric_columns)

            if st.button("Compare"):
                group_and_compare(df1, df2, groupby_columns, selected_metrics)
        else:
            st.error("No common columns found between the files.")
