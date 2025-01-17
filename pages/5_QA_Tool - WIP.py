# ---------------------------------- Import Libraries ---------------------------------- #
import streamlit as st
import pandas as pd
# from rapidfuzz import process, fuzz # might not be needed if we are moving forward with Levenshtein
# import spacy # might not be needed if we are moving forward with Levenshtein
import re  # regex
import Levenshtein

# Define the list of default columns to be included in the group by session
default_groupby_columns = ['placement_id','campaign_id', "creative_id", 'site_id']

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
        st.error("Unsupported file type. Please try uploading a CSV or Excel file.")
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
            if col in used_matches: # Avoid duplicating matches
                continue 

            distance = Levenshtein.distance(input_col_lower, col) # Calculate the Levenshtein distance
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

def convert_date_columns(df):
    """Convert date related columns to datetime format"""
    # Need to confirm if there are any issues that could happen with different date formats
    date_keywords = ['date', 'week', 'month', 'year'] # Keywords to identify date columns

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
        # st.success("Files uploaded successfully")
        df1 = convert_date_columns(df1)
        df2 = convert_date_columns(df2)

        if st.button("View Data"):
            st.write("### File 1:")
            st.write(df1.head())
            st.write("### File 2:")
            st.write(df2.head())
        return df1, df2
    return None, None

def auto_match_columns(df1, df2):
    """Automatically match and rename columns between two dataframes."""
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)

    st.write("### Column Matching")
    auto_match = st.checkbox("Enable automatic column renaming", value=True)
    column_replacements = {}

    if auto_match:
        st.success("The automatic column renaming feature is enabled. Click the toggle above to undo the changes.")
        matches = find_best_match(columns_2, columns_1)
        for col, matched_col in matches.items():
            if matched_col != col:
                column_replacements[col] = matched_col
        df2.rename(columns=column_replacements, inplace=True)
        
        if not column_replacements:
            st.info("No columns were automatically renamed. The files may already have matching columns.")
    else:
        st.info("Automatic column renaming is disabled. Column names will remain as uploaded.")

    return df1, df2, column_replacements 

def manual_edit_columns(df1, df2):
    """Allow manual editing of column names for both dataframes."""
    st.markdown("---")
    st.write("### Manual Column Editing")

    # Create a dataframe to display column names side by side
    max_len = max(len(df1.columns), len(df2.columns))
    columns_df = pd.DataFrame({
        "File 1 Columns": list(df1.columns) + [""] * (max_len - len(df1.columns)),
        "File 2 Columns": list(df2.columns) + [""] * (max_len - len(df2.columns))
    })

    # Toggle to show/hide the table
    show_columns_table = st.checkbox("Show Column Names Table")

    # Display table, if chosen
    if show_columns_table:
        # custom CSS to fix column names aligned to the right
        st.markdown(
            """
            <style>
            .dataframe th {
                text-align: left !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display the dataframe as a table without the index
        st.write("#### Column Names in File 1 and File 2")
        st.write(columns_df.to_html(index=False), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("##### Select Columns to Rename in File 1")
        columns_to_rename_file1 = st.multiselect("Select columns from File 1:", df1.columns)

    with col2:
        st.write("##### Select Columns to Rename in File 2")
        columns_to_rename_file2 = st.multiselect("Select columns from File 2:", df2.columns)

    # Show text input for selected columns in File 1
    if columns_to_rename_file1:
        st.write("#### Rename Selected Columns in File 1")
        for col in columns_to_rename_file1:
            new_col = st.text_input(f"Rename '{col}'", col, key=f"file1_{col}")
            df1.rename(columns={col: new_col}, inplace=True)

    # Show text input for selected columns in File 2
    if columns_to_rename_file2:
        st.write("#### Rename Selected Columns in File 2")
        for col in columns_to_rename_file2:
            new_col = st.text_input(f"Rename '{col}'", col, key=f"file2_{col}")
            df2.rename(columns={col: new_col}, inplace=True)

    return df1, df2

def group_and_compare(df1, df2, groupby_columns, selected_metrics):
    """Group dataframe and compare metrics."""
    # Identify datetime columns in the dataframe
    datetime_columns_1 = df1.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    datetime_columns_2 = df2.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

    # rename columns to distinguish between files
    df1.columns = [f"{col} - File 1" if col not in groupby_columns else col for col in df1.columns]
    df2.columns = [f"{col} - File 2" if col not in groupby_columns else col for col in df2.columns]

    # Ensure selected metrics in the dataframe
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
    discrepancy_mask = pd.Series([False] * len(merged_df))

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

            # Update the discrepancy baseline
            discrepancy_mask |= merged_df[pct_diff_col].abs() > 0.5

    # Add a column for discrepancies
    results['Discrepancy > 0.5%'] = discrepancy_mask

    st.write("#### Side by Side Comparison")
    st.write(results)

    if discrepancy_mask.any():
        significant_discrepancies = results[discrepancy_mask]

        # Calculate discrepancy percentage based on grouped rows
        grouped_total_rows = len(merged_df)
        num_discrepancies = len(significant_discrepancies.drop_duplicates(subset=groupby_columns))
        discrepancy_percentage = (num_discrepancies / grouped_total_rows) * 100

        st.error(f"Found {num_discrepancies} rows with discrepancies larger than 0.5%, representing {discrepancy_percentage:.2f}% of the total grouped data.")
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

    st.markdown("---")
    
    if df1 is not None and df2 is not None:
        df1, df2, column_replacements = auto_match_columns(df1, df2)

        # Display renamed columns
        if column_replacements:
            # st.write("### Renamed Columns in File 2")
            st.write("The following columns in File 2 were automatically renamed to align with the column names in File 1:")
            for old_col, new_col in column_replacements.items():
                st.write(f"**'{old_col}'** was renamed to **'{new_col}'**")

        # Toggle for showing/hiding manual edit options
        if "show_manual_edit" not in st.session_state:
            st.session_state.show_manual_edit = False
            # st.markdown("---")

        if st.button("Edit Column Names"):
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
            
            # Filter default columns to only include those that exist in groupby_options
            valid_default_groupby_columns = [col for col in default_groupby_columns if col in groupby_options]
            
            # Pre-select valid default columns and allow users to add more
            groupby_columns = st.multiselect(
                "Select columns to group by:",
                groupby_options,
                default=valid_default_groupby_columns
            )
            
            # Ensure no duplicates in final group-by list
            groupby_columns = list(set(valid_default_groupby_columns) | set(groupby_columns))

            numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(df1[col])]
            selected_metrics = st.multiselect("Select metrics to compare:", numeric_columns)

            if st.button("Compare"):
                group_and_compare(df1, df2, groupby_columns, selected_metrics)
        else:
            st.error("No common columns found between the files.")
            st.error("You can edit the column names by clicking the 'Edit Column Names' button above.")

                