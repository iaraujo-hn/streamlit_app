# ---------------------------------- Import Libraries ---------------------------------- #
import streamlit as st
import pandas as pd
import re  # regex
import Levenshtein

# define the list of default columns to be included in the group by session
default_groupby_columns = ['placement_id','campaign_id', "creative_id", 'site_id']

# columns to exclude from the group by selection
excluded_groupby_columns = [
    'impressions', 'clicks', 'Click', 'spend', 'sessions', 'page_views', 'revenue', 'conversions'
]

# ---------------------------------- File Upload ---------------------------------- #
def clean_id_columns(df):
    """
    Fixes issue where numeric categorical columns from csv were adding an extra decimal place, causing mismatches.
    - Converts numeric values to integers (removing decimals)
    - Converts all values to strings to ensure consistency
    """
    for col in df.columns:
        if "id" in col.lower():  # Check if id is in the column name
            if pd.api.types.is_numeric_dtype(df[col]):
                # Remove decimals by converting to integers and then to strings
                df[col] = df[col].fillna(0).apply(lambda x: str(int(x)) if not pd.isnull(x) else "0")
            else:
                # Convert non numeric values directly to strings
                df[col] = df[col].astype(str)
    return df

# @st.cache_data
# def read_file(file):
#     """Read csv or excel file based on file extension"""
#     with st.spinner('Loading file...'):
#         if file.name.endswith('.csv'):
#             return pd.read_csv(file)
#         elif file.name.endswith('.xlsx'):
#             return pd.read_excel(file, engine='openpyxl')  # adding engine to fix the issue where large files were throwing an error
#         else:
#             st.error("Unsupported file type. Please try uploading a CSV or Excel file.")
#             return None

@st.cache_data
def read_file(file):
    """Read csv or excel file based on file extension and clean ID columns."""
    with st.spinner('Loading file...'):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        
        # Clean ID columns
        df = clean_id_columns(df)
        return df


def find_best_match(input_cols, columns_list):
    """Find the best match for each input column using Levenshtein distance, with improved logic to avoid mismatches."""
    input_cols_lower = [col.lower() for col in input_cols]
    columns_list_lower = [col.lower() for col in columns_list]
    used_matches = set()
    best_matches = {}

    for input_col, input_col_lower in zip(input_cols, input_cols_lower):
        best_match = None
        best_distance = float('inf')

        # prioritize exact matches
        for i, col in enumerate(columns_list_lower):
            if col in used_matches:  # avoid duplications
                continue

            if col == input_col_lower:  # exact match
                best_match = columns_list[i]
                best_distance = 0
                break

        # apply levenshtein distance only if no exact match found
        if not best_match:
            for i, col in enumerate(columns_list_lower):
                if col in used_matches:  # avoid duplications
                    continue

                distance = Levenshtein.distance(input_col_lower, col)

                # avoid swapping similar but distinct columns
                is_conflicting = (
                    (input_col_lower in col or col in input_col_lower)
                    and ("id" in input_col_lower and "id" not in col or "id" not in input_col_lower and "id" in col) # adds extra protection for columns with id in the name
                )

                if distance < best_distance and not is_conflicting:
                    best_match = columns_list[i]
                    best_distance = distance

        # only assign a match if it meets the threshold and does not overwrite meaningful differences
        if best_match and best_distance <= 2:  # threshold for levenshtein distance. This is how many fixes are needed to match the strings
            best_matches[input_col] = best_match
            used_matches.add(best_match.lower())
        else:
            best_matches[input_col] = input_col  # no suitable match found. retain original name

    return best_matches


def convert_date_columns(df):
    """Convert date related columns to datetime format"""
    # Need to confirm if there are any issues that could happen with different date formats
    date_keywords = ['date', 'month', 'year'] # keywords to identify date columns. removed week as it was throwing off the date conversion with week number columns

    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            sample_values = df[col].dropna().head(10).astype(str)
            date_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}') # regex for date format

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

# def clean_id_columns(df):
#     """
#     Adds extra protection for columns with 'ID' in the name:
#     - Converts numeric values to integers (removing decimals).
#     - Converts all values to strings.
#     """
#     for col in df.columns:
#         if "id" in col.lower():  # check if id is in the column name
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 # convert numeric values to integers, then to strings
#                 df[col] = df[col].fillna(0).astype(int).astype(str)
#             else:
#                 # convert non-numeric values directly to strings
#                 df[col] = df[col].astype(str)
#     return df

def clean_id_columns(df):
    """
    Cleans columns with 'ID' in their name:
    - Converts numeric values to integers (removing decimals).
    - Converts all values to strings to ensure consistency.
    """
    for col in df.columns:
        if "id" in col.lower():  # Check if 'id' is in the column name
            if pd.api.types.is_numeric_dtype(df[col]):
                # Remove decimals by converting to integers and then to strings
                df[col] = df[col].fillna(0).apply(lambda x: str(int(x)) if not pd.isnull(x) else "0")
            else:
                # Convert non-numeric values directly to strings
                df[col] = df[col].astype(str)
    return df

# Update the display_uploaded_data function to include clean_id_columns
# @st.cache_data
# def display_uploaded_data(file_1, file_2):
#     """
#     Load and preprocess uploaded files
#     Ensures groupby columns are treated as categorical.
#     """
#     df1 = read_file(file_1)
#     df2 = read_file(file_2)

#     if df1 is not None and df2 is not None:
#         # convert date columns to datetime (if needed)
#         df1 = convert_date_columns(df1)
#         df2 = convert_date_columns(df2)

#         # clean and protect ID columns
#         df1 = clean_id_columns(df1)
#         df2 = clean_id_columns(df2)

#         # ensure default group by columns are treated as strings
#         for col in default_groupby_columns:
#             if col in df1.columns:
#                 df1[col] = df1[col].astype(str)
#             if col in df2.columns:
#                 df2[col] = df2[col].astype(str)

#         return df1, df2

#     return None, None

# def display_uploaded_data(file_1, file_2):
#     """Upload and display data from two files."""
#     df1 = read_file(file_1)
#     df2 = read_file(file_2)

#     if df1 is not None and df2 is not None:
#         # st.success("Files uploaded successfully")
#         df1 = convert_date_columns(df1)
#         df2 = convert_date_columns(df2)

#         if st.button("View Data"):
#             st.write("### File 1:")
#             st.write(df1.head())
#             st.write("### File 2:")
#             st.write(df2.head())
#         return df1, df2
#     return None, None

def display_uploaded_data(file_1, file_2):
    """Upload and display data from two files with toggling and tabs."""
    df1 = read_file(file_1)
    df2 = read_file(file_2)

    if df1 is not None and df2 is not None:
        # Process the data
        df1 = convert_date_columns(df1)
        df2 = convert_date_columns(df2)

        # Create session state variables for toggling
        if "show_data" not in st.session_state:
            st.session_state.show_data = False

        # Toggle button for showing/hiding data
        if st.button("View Data"):
            st.session_state.show_data = not st.session_state.show_data

        # Conditionally display the data
        if st.session_state.show_data:
            st.write("### Uploaded Data:")
            tab1, tab2 = st.tabs(["File 1", "File 2"])

            with tab1:
                st.write("### File 1:")
                st.write(df1)

            with tab2:
                st.write("### File 2:")
                st.write(df2)

        return df1, df2
    return None, None
    

def auto_match_columns(df1, df2):
    """
    Automatically match and rename columns between two dataframes
    Ensures default group by columns are treated as categorical
    """
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)

    st.write("### Column Matching")
    auto_match = st.checkbox("Enable automatic column renaming", value=True)
    column_replacements = {}

    if auto_match:
        matches = find_best_match(columns_2, columns_1)
        for col, matched_col in matches.items():
            if matched_col != col:
                column_replacements[col] = matched_col
        df2.rename(columns=column_replacements, inplace=True)

        if not column_replacements:
            st.info("No columns were automatically renamed. The files may already have matching columns.")
    else:
        st.info("Automatic column renaming is disabled. Column names will remain as uploaded.")

    # Ensure default group by columns are categorical
    for col in default_groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    return df1, df2, column_replacements

def manual_edit_columns(df1, df2):
    """
    Allow manual editing of column names and data types for both dataframes
    Display side by side input boxes when renaming columns in both files
    Ensure the maximum number of rows is shown, filling missing selections with placeholders
    """
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

    # Select columns to rename in both files
    col1, col2 = st.columns(2)

    with col1:
        st.write("##### Select Columns to Rename in File 1")
        columns_to_rename_file1 = st.multiselect("Select columns from File 1:", df1.columns)

    with col2:
        st.write("##### Select Columns to Rename in File 2")
        columns_to_rename_file2 = st.multiselect("Select columns from File 2:", df2.columns)

    # Determine the maximum number of columns selected
    max_columns_to_rename = max(len(columns_to_rename_file1), len(columns_to_rename_file2))

    # Extend the shorter list with placeholders
    extended_file1_columns = columns_to_rename_file1 + ["[No Selection]"] * (max_columns_to_rename - len(columns_to_rename_file1))
    extended_file2_columns = columns_to_rename_file2 + ["[No Selection]"] * (max_columns_to_rename - len(columns_to_rename_file2))


    if max_columns_to_rename > 0:
        st.write(f"##### Rename Selected Columns")

        # Loop through the extended lists
        for col1_name, col2_name in zip(extended_file1_columns, extended_file2_columns):
            col1, col2 = st.columns(2)

            # Input field for File 1
            with col1:
                if col1_name != "[No Selection]":
                    new_col_name1 = st.text_input(
                        f"Rename '{col1_name}' (File 1):", col1_name, key=f"rename_file1_{col1_name}"
                    )
                    df1.rename(columns={col1_name: new_col_name1}, inplace=True)
                else:
                    st.write(" ")

            # Input field for File 2
            with col2:
                if col2_name != "[No Selection]":
                    new_col_name2 = st.text_input(
                        f"Rename '{col2_name}' (File 2):", col2_name, key=f"rename_file2_{col2_name}"
                    )
                    df2.rename(columns={col2_name: new_col_name2}, inplace=True)
                else:
                    st.write(" ")
    return df1, df2

def prepare_group_and_metric_options(df1, df2, default_groupby_columns, excluded_groupby_columns):
    """
    Prepare group by and metrics options:
    - Ensure default groupby columns are not preselected in the group by dropdown if they contain "ID".
    - Treat default groupby columns as categorical.
    - Exclude default groupby columns from metrics options.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))

    # Group by options: Exclude columns in the excluded groupby list
    groupby_options = sorted([col for col in common_columns if col not in excluded_groupby_columns])

    # Filter default groupby columns to exclude those with "ID" in their name
    valid_default_groupby_columns = [
        col for col in default_groupby_columns
        if col in groupby_options and "id" not in col.lower()
    ]

    # Ensure default group by columns are treated as categorical
    for col in default_groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    # Numeric metrics options: Exclude groupby columns
    numeric_columns = [
        col for col in common_columns
        if pd.api.types.is_numeric_dtype(df1[col]) and col not in valid_default_groupby_columns
    ]

    return groupby_options, valid_default_groupby_columns, numeric_columns



def clean_and_convert_to_numeric(series):
    """
    Clean and convert a pandas Series to numeric 
    Handles cases with commas, dollar signs, and other non numeric characters 
    Converts to int if no decimal point exists, otherwise float 
    """
    # remove non numeric characters except for periods to fix currency format issues
    cleaned_series = (
        series.astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)  # remove everything except numbers and periods
        .str.replace(r"\.+", ".", regex=True)  # handle multiple periods, keeping only one
    )
    # Convert to numeric, coercing errors
    return pd.to_numeric(cleaned_series, errors="coerce")


def ensure_numeric_and_clean(df, numeric_columns):
    """
    Ensure that specified columns in the DataFrame are numeric (int or float)
    Skip columns in default groupby columns to prevent them from being converted
    """
    for col in numeric_columns:
        # Skip columns in default groupby columns
        if col in default_groupby_columns:
            continue
        if col in df.columns:
            df[col] = clean_and_convert_to_numeric(df[col])
    return df

def group_and_compare(df1, df2, groupby_columns, selected_metrics):
    """Group dataframes and compare metrics."""
    # Ensure groupby columns in default groupby columns are treated as strings
    for col in default_groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    # Ensure groupby columns have consistent types across both dataframes
    for col in groupby_columns:
        if col in df1.columns and col in df2.columns:
            df1[col] = df1[col].astype(str)
            df2[col] = df2[col].astype(str)

    # Ensure numeric columns are converted to numeric types
    # Clean and ensure numeric columns, excluding default group by columns
    df1 = ensure_numeric_and_clean(df1, [col for col in selected_metrics if col not in default_groupby_columns])
    df2 = ensure_numeric_and_clean(df2, [col for col in selected_metrics if col not in default_groupby_columns])

    # Rename columns to distinguish between files, but only for selected groupby columns
    df1.columns = [f"{col} - File 1" if col not in groupby_columns else col for col in df1.columns]
    df2.columns = [f"{col} - File 2" if col not in groupby_columns else col for col in df2.columns]

    # Group by the specified columns and sum the selected metrics
    df1_grouped = df1.groupby(groupby_columns)[[f"{col} - File 1" for col in selected_metrics]].sum().reset_index()
    df2_grouped = df2.groupby(groupby_columns)[[f"{col} - File 2" for col in selected_metrics]].sum().reset_index()

    # Merge the grouped dataframes
    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns, how="outer")

    # Prepare results DataFrame for output
    results = merged_df[groupby_columns].copy() if groupby_columns else pd.DataFrame()
    discrepancy_mask = pd.Series([False] * len(merged_df))

    for col in selected_metrics:
        col_A = f"{col} - File 1"
        col_B = f"{col} - File 2"
        diff_col = f"{col} Difference"
        pct_diff_col = f"{col} % Difference"

        if col_A in merged_df.columns and col_B in merged_df.columns:
            # Calculate differences
            merged_df[diff_col] = merged_df[col_A] - merged_df[col_B]
            merged_df[pct_diff_col] = (merged_df[diff_col] / merged_df[col_B]) * 100

            # Format the percentage difference
            results[col_A] = merged_df[col_A]
            results[col_B] = merged_df[col_B]
            results[diff_col] = merged_df[diff_col]
            results[pct_diff_col] = merged_df[pct_diff_col].apply(lambda x: f"{x:.2f}%")

            # Identify rows with significant discrepancies
            discrepancy_mask |= merged_df[pct_diff_col].abs() > 0.5

    # Add a column for discrepancies
    if not results.empty:
        results["Sig. Discrepancy"] = discrepancy_mask.apply(lambda x: "Yes" if x else "No")

    st.write("#### Side by Side Comparison")
    st.write(results)

    if discrepancy_mask.any():
        significant_discrepancies = results[discrepancy_mask]

        # Calculate discrepancy percentage
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
            st.write("The following columns in File 2 were automatically renamed to align with the column names in File 1:")
            for old_col, new_col in column_replacements.items():
                st.write(f"**'{old_col}'** was renamed to **'{new_col}'**")

        # Toggle for showing/hiding manual edit options
        if "show_manual_edit" not in st.session_state:
            st.session_state.show_manual_edit = False

        if st.button("Edit Column Names"):
            st.session_state.show_manual_edit = not st.session_state.show_manual_edit

        # Conditionally display the manual edit options
        if st.session_state.show_manual_edit:
            df1, df2 = manual_edit_columns(df1, df2)

        st.markdown("---")

        # Prepare group by and metrics options using the new helper function
        groupby_options, valid_default_groupby_columns, numeric_columns = prepare_group_and_metric_options(
            df1, df2, default_groupby_columns, excluded_groupby_columns
        )

        # UI for selecting group by columns
        groupby_columns = st.multiselect(
            "Select columns to group by:",
            groupby_options,
            default=valid_default_groupby_columns
        )

        # UI for selecting metrics
        selected_metrics = st.multiselect("Select metrics to compare:", numeric_columns)

        if st.button("Compare"):
            group_and_compare(df1, df2, groupby_columns, selected_metrics)



                