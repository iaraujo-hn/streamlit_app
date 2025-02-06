# ---------------------------------- Import Libraries ---------------------------------- #
import streamlit as st
import pandas as pd
import re  # regex
import chardet
import Levenshtein

# ---------------------------------- Dictionaries ---------------------------------- #
# Columns to be included in the group by selection only
default_groupby_columns = ['placement_id','campaign_id', "creative_id", 'site_id']

# Columns to exclude from the group by selection
excluded_groupby_columns = [
    'impressions', 'Impressions','clicks', 'Clicks', 'link_click','spend', 'Spend','Cost','Media Cost','sessions', 'page_views', 'revenue', 'conversions'
]

# Common column mappings for automatic matching
column_mapping = {
    "site_dcm": "Site (CM360)",
    "Site ID (CM360)": "site_id_dcm",
    "spend": "Media Cost",
    "Cost": "spend"
}

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

@st.cache_data
def read_file(file):
    """Read CSV or Excel file with automatic encoding detection and error handling, including skipping metadata rows."""
    with st.spinner('Loading file...'):
        df = None
        error_message = None

        try:
            if file.name.endswith('.csv'):
                raw_bytes = file.read(10000)  # Read a portion of the file
                detected_encoding = chardet.detect(raw_bytes)['encoding']  # Detect encoding
                file.seek(0)  # Reset file pointer

                # Try reading the file, with a fallback for skipping metadata rows
                for skip_rows in range(11):  # Try skipping from 0 to 10 rows
                    try:
                        df = pd.read_csv(file, encoding=detected_encoding if detected_encoding else 'utf-8', skiprows=skip_rows)
                        break  # Stop if successful
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        file.seek(0)  # Reset file pointer before retrying
                
                if df is None:
                    raise pd.errors.ParserError("Failed to read file after skipping 10 rows.")

            elif file.name.endswith('.xlsx'):
                for skip_rows in range(11):
                    try:
                        df = pd.read_excel(file, engine='openpyxl', skiprows=skip_rows)
                        break
                    except Exception:
                        file.seek(0)
                
                if df is None:
                    raise Exception("Failed to read Excel file after skipping 10 rows.")

            else:
                st.error("❌ Unsupported file type. Please upload a **CSV or Excel file**.")
                return None

            # Validate if the dataframe has proper headers and data
            if df.empty or df.columns.notna().sum() <= 1:
                error_message = "⚠️ The file seems to be **empty or incorrectly formatted**. Please check that it contains valid data."

        except UnicodeDecodeError:
            error_message = "⚠️ There was an issue reading the file due to **invalid characters**. Try saving it again using UTF-8 encoding."
        except pd.errors.EmptyDataError:
            error_message = "⚠️ The file appears to be **empty**. Please verify its contents."
        except pd.errors.ParserError:
            error_message = "⚠️ The file format may be **corrupted or incorrect**, or it contains too many metadata rows. Try checking its structure."
        except Exception as e:
            error_message = f"⚠️ Unexpected error: {str(e)}"
        
        # If any error was detected, display it and troubleshooting tips
        if error_message:
            st.error(error_message)
            st.markdown("### How to Fix This:")
            st.markdown("""
            - **Check if the file contains metadata in the first few rows** and remove them manually.
            - **Ensure it's in CSV or Excel format:** Other file types (e.g., PDF, Word) are not supported.
            - **If you see character issues, save the file using UTF-8 encoding.**  
              - In Excel: Save As → Choose CSV (UTF-8)
              - In Notepad: File → Save As → Encoding: UTF-8
            - **If it's an Excel file, try saving it again as a new Excel file:** This can fix formatting issues.
            - **If the issue persists, try uploading a different version of the file.**
            """)
            return None
        
        # Clean ID columns after successful load
        df = clean_id_columns(df)
        return df

def preprocess_column_for_matching(column):
    """
    Preprocess a column name by removing 'name' and converting to lowercase.
    Used to test if removing 'name' results in a match.
    """
    return re.sub(r'\bname\b', '', column.lower()).strip()

import re
import Levenshtein

def preprocess_column_for_matching(column):
    """
    Preprocess a column name by removing 'name' and converting to lowercase.
    Used to test if removing 'name' results in a match.
    """
    return re.sub(r'\bname\b', '', column.lower()).strip()

def find_best_match(input_cols, columns_list):
    """
    Find the best match for each input column using Levenshtein distance.
    - If removing 'name' from both columns results in a match, apply the match.
    - If removing 'name' doesn't result in a match, use standard Levenshtein matching.
    """
    used_matches = set()
    best_matches = {}

    # Preprocess column names by removing 'name'
    processed_input_cols = {col: preprocess_column_for_matching(col) for col in input_cols}
    processed_columns_list = {col: preprocess_column_for_matching(col) for col in columns_list}

    for input_col in input_cols:
        best_match = None
        best_distance = float('inf')

        # Preprocessed version of input column
        input_col_processed = processed_input_cols[input_col]

        # Try exact match after removing "name"
        for col in columns_list:
            if col in used_matches:
                continue

            if input_col_processed == processed_columns_list[col]:
                best_match = col
                best_distance = 0
                break

        # Apply Levenshtein distance if no exact match is found
        if not best_match:
            for col in columns_list:
                if col in used_matches:
                    continue

                # Compute Levenshtein distance after preprocessing (removing 'name')
                distance = Levenshtein.distance(input_col_processed, processed_columns_list[col])

                # Avoid incorrect matches
                is_conflicting = (
                    (input_col.lower() in col.lower() or col.lower() in input_col.lower()) and
                    ("id" in input_col.lower() and "id" not in col.lower() or "id" not in input_col.lower() and "id" in col.lower())
                )

                if distance < best_distance and not is_conflicting:
                    best_match = col
                    best_distance = distance

        # Assign a match only if it meets the threshold and does not create conflicts
        if best_match and best_distance <= 3:  # Allow small differences for typos or variations. Adjust as needed
            best_matches[input_col] = best_match
            used_matches.add(best_match)
        else:
            best_matches[input_col] = input_col  # No suitable match found, retain original name

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

        # Toggle button for showing or hiding data
        if st.button("View Data"):
            st.session_state.show_data = not st.session_state.show_data

        # Conditionally display the data
        if st.session_state.show_data:
            tab1, tab2 = st.tabs(["Main File", "File 2"])

            with tab1:
                st.write(df1.head(10))
                st.write('*Displaying the first 10 rows only.')

            with tab2:
                st.write(df2.head(10))
                st.write('*Displaying the first 10 rows only.')

        return df1, df2
    return None, None

def auto_match_columns(df1, df2, column_mapping):
    """
    Automatically match and rename columns between two dataframes
    Ensures specific columns are renamed only if their counterpart exists in the other file.
    """
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)

    column_replacements = {}

    # Identify renaming candidates
    for col1, col2 in column_mapping.items():
        if col1 in columns_1 and col2 in columns_2:
            # Rename column in file 2 to match main file
            column_replacements[col2] = col1
        elif col2 in columns_1 and col1 in columns_2:
            # Rename column in main file to match file 2
            column_replacements[col1] = col2

    # Apply renaming
    df1.rename(columns=column_replacements, inplace=True)
    df2.rename(columns=column_replacements, inplace=True)

    # Regular auto matching process
    matches = find_best_match(columns_2, columns_1)
    for col, matched_col in matches.items():
        if matched_col != col and col not in column_replacements:
            column_replacements[col] = matched_col

    df2.rename(columns=column_replacements, inplace=True)

    return df1, df2, column_replacements

# ---------------------------------- UI Components ---------------------------------- #

def manual_edit_columns(df1, df2, column_replacements_auto):
    """
    Allow manual editing of column names in File 2 to match main file.
    Display column names side by side, placing matching columns at the top.
    Ensure main file remains unchanged.
    """
    # Identify matching and non-matching columns
    matching_columns = [col for col in df1.columns if col in df2.columns]
    non_matching_columns_1 = [col for col in df1.columns if col not in df2.columns]
    non_matching_columns_2 = [col for col in df2.columns if col not in df1.columns]

    # Order the dataframe: matches first, then non-matches in their original order
    ordered_columns_1 = matching_columns + non_matching_columns_1
    ordered_columns_2 = matching_columns + non_matching_columns_2

    # Create dataframe for UI display
    max_len = max(len(ordered_columns_1), len(ordered_columns_2))
    columns_df = pd.DataFrame({
        "Main File": ordered_columns_1 + [""] * (max_len - len(ordered_columns_1)),
        "File 2": ordered_columns_2 + [""] * (max_len - len(ordered_columns_2))
    })

    # Function to highlight matching columns
    def highlight_matching_cols(val):
        if val in matching_columns:
            return "background-color: #c6f5c6; font-weight: bold;"  # Green highlight
        return ""

    # Toggle to show/hide the table
    show_columns_table = st.checkbox("Show column names")

    if show_columns_table:
        # Column name styling
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
                white-space: nowrap;
            }
            tbody td {
                padding: 12px;
                white-space: normal !important;
                word-wrap: break-word !important;
                max-width: 600px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        styled_df = (
            columns_df.style
            .applymap(highlight_matching_cols)
            .set_properties(**{"max-width": "600px", "white-space": "normal"})
        )

        st.write("#### Column Names")
        st.write("Columns in green match between Main File and File 2.")
        
        if column_replacements_auto:
            renamed_columns_tooltip = "\n".join([f"{old} → {new}" for old, new in column_replacements_auto.items()])
            tooltip_text = f"""
            <div style="border-bottom: 1px dotted; display: inline; cursor: help;" title="{renamed_columns_tooltip}">
            ℹ️ Automatically Renamed Columns (hover to view)
            </div>
            """
        st.markdown(tooltip_text, unsafe_allow_html=True)

        st.table(styled_df)

    # Track manual renaming
    column_replacements = {}

    st.write("##### Select Columns to Rename in File 2")
    columns_to_rename_file2 = st.multiselect("Select columns from File 2:", df2.columns)

    if columns_to_rename_file2:
        st.write("##### Rename Selected Columns in File 2 to Match Main File")

        for col2_name in columns_to_rename_file2:
            col1, _ = st.columns([2, 3])

            with col1:
                new_col_name2 = st.selectbox(
                    f"Rename '{col2_name}' (File 2):",
                    [""] + list(df1.columns),
                    key=f"rename_file2_{col2_name}"
                )

                if new_col_name2 and col2_name != new_col_name2:
                    column_replacements[col2_name] = new_col_name2
                    df2.rename(columns={col2_name: new_col_name2}, inplace=True)

    final_matching = sum(1 for col in df2.columns if col in df1.columns)

    if column_replacements:
        for old_col, new_col in column_replacements.items():
            st.write(f"**'{old_col}'** renamed to **'{new_col}'**")
        st.success(f"✅ {final_matching} out of {len(df1.columns)} columns in File 2 now match Main File.")

    return df1, df2, column_replacements

def prepare_group_and_metric_options(df1, df2, default_groupby_columns, excluded_groupby_columns):
    """
    Prepare group by and metrics options:
    - Ensure default groupby columns are not preselected in the group by dropdown if they contain "ID".
    - Treat default groupby columns as categorical.
    - Exclude default groupby columns from metrics options.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))

    # Group by options: exclude columns in the excluded groupby list
    groupby_options = sorted([col for col in common_columns if col not in excluded_groupby_columns])

    # Filter default groupby columns to exclude those with ID in their name
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

    # Numeric metrics options: exclude groupby columns
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
    """Group dataframes and compare metrics, ensuring 0,0 pairs are included in the comparison."""
    for col in default_groupby_columns:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    for col in groupby_columns:
        if col in df1.columns and col in df2.columns:
            df1[col] = df1[col].astype(str)
            df2[col] = df2[col].astype(str)

    df1 = ensure_numeric_and_clean(df1, [col for col in selected_metrics if col not in default_groupby_columns])
    df2 = ensure_numeric_and_clean(df2, [col for col in selected_metrics if col not in default_groupby_columns])

    df1.columns = [f"{col} - Main File" if col not in groupby_columns else col for col in df1.columns]
    df2.columns = [f"{col} - File 2" if col not in groupby_columns else col for col in df2.columns]

    df1_grouped = df1.groupby(groupby_columns)[[f"{col} - Main File" for col in selected_metrics]].sum().reset_index()
    df2_grouped = df2.groupby(groupby_columns)[[f"{col} - File 2" for col in selected_metrics]].sum().reset_index()

    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns, how="outer").fillna(0)

    results = merged_df[groupby_columns].copy() if groupby_columns else pd.DataFrame()
    discrepancy_mask = pd.Series([False] * len(merged_df))

    for col in selected_metrics:
        col_A = f"{col} - Main File"
        col_B = f"{col} - File 2"
        diff_col = f"{col} Difference"
        pct_diff_col = f"{col} % Difference"

        if col_A in merged_df.columns and col_B in merged_df.columns:
            merged_df[diff_col] = merged_df[col_A] - merged_df[col_B]
            
            # Handle division by zero properly
            def calculate_percentage_difference(a, b):
                if a == 0 and b == 0:
                    return 0  # Both are zero, difference is 0%
                elif b == 0:
                    return float('inf') if a != 0 else 0  # Avoid division by zero issues
                else:
                    return (a - b) / abs(b) * 100

            merged_df[pct_diff_col] = merged_df.apply(lambda row: calculate_percentage_difference(row[col_A], row[col_B]), axis=1)

            results[col_A] = merged_df[col_A].fillna(0)
            results[col_B] = merged_df[col_B].fillna(0)
            results[diff_col] = merged_df[diff_col]
            results[pct_diff_col] = merged_df[pct_diff_col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else "∞%")

            discrepancy_mask |= merged_df[pct_diff_col].abs() > 0.5

    if not results.empty:
        results["Sig. Discrepancy"] = discrepancy_mask.apply(lambda x: "Yes" if x else "No")

    # Store results persistently in session state
    st.session_state["comparison_results"] = results
    st.session_state["discrepancy_mask"] = discrepancy_mask
    st.session_state["groupby_columns"] = groupby_columns  # Store for later use

    # Always display the main comparison table
    st.write("### Side-by-Side Comparison")
    st.write(results)

    # Display discrepancy summary above the toggle
    if discrepancy_mask.any():
        significant_discrepancies = results[discrepancy_mask]
        grouped_total_rows = len(results)
        num_discrepancies = len(significant_discrepancies.drop_duplicates(subset=groupby_columns))
        discrepancy_percentage = (num_discrepancies / grouped_total_rows) * 100

        st.error(f"Found {num_discrepancies} rows with discrepancies larger than 0.5%, representing {discrepancy_percentage:.2f}% of the total grouped data.")
    else:
        st.success("✅ No discrepancies greater than 0.5% found.")

    # Download button for results table
    if not results.empty:
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="comparison_results.csv",
            mime="text/csv",
            key="download_results"
        )

def display_discrepancies():
    """Displays discrepancies independently from the comparison function."""
    if "comparison_results" not in st.session_state or "discrepancy_mask" not in st.session_state:
        return  # do nothing if results don't exist yet

    results = st.session_state["comparison_results"]
    discrepancy_mask = st.session_state["discrepancy_mask"]

    # Only show the toggle button if there are discrepancies
    if discrepancy_mask.any():
        show_discrepancy_table = st.checkbox("Show discrepancy table")

        if show_discrepancy_table:
            significant_discrepancies = results[discrepancy_mask]
            st.write("### Significant Discrepancies")
            st.write("Showing rows with discrepancies larger than 0.5%. Click on **Compare** again to see the full comparison.")
            st.write(significant_discrepancies)

def check_and_convert_to_numeric(df, col):
    """Convert column to numeric, coercing errors."""
    return pd.to_numeric(df[col], errors='coerce')

# ---------------------------------- Main App ---------------------------------- #
st.title("QA Tool")
st.write("The QA Tool compares data between two files, highlighting discrepancies.")

file_1 = st.file_uploader("**Choose the first file:**", type=["csv", "xlsx"], key="file_1")
file_2 = st.file_uploader("**Choose the second file:**", type=["csv", "xlsx"], key="file_2")

if file_1 and file_2:
    df1, df2 = display_uploaded_data(file_1, file_2)

    st.markdown("---")

    if df1 is not None and df2 is not None:
        df1, df2, column_replacements_auto = auto_match_columns(df1, df2, column_mapping)

        total_columns = len(df1.columns)
        matching_after_auto = sum(1 for col in df2.columns if col in df1.columns)

        st.write("#### Columns Editing")
        st.write("QA Tool automatically renames similar columns since only matching names can be compared.")

        if matching_after_auto == total_columns:
            st.success(f"✅ All {total_columns} columns in File 2 match Main File!")
        elif matching_after_auto > 0:
            st.info(f"ℹ️ {matching_after_auto} out of {total_columns} columns in File 2 match Main File.")
        else:
            st.warning("⚠️ None of the columns in File 2 match Main File. Consider renaming manually.")

        # Toggle for manual edit options
        if "show_manual_edit" not in st.session_state:
            st.session_state.show_manual_edit = False

        if st.button("Edit Column Names"):
            st.session_state.show_manual_edit = not st.session_state.show_manual_edit

        if st.session_state.show_manual_edit:
            df1, df2, column_replacements_manual = manual_edit_columns(df1, df2, column_replacements_auto)

        st.markdown("---")

        # Prepare group by and metrics options
        groupby_options, valid_default_groupby_columns, numeric_columns = prepare_group_and_metric_options(
            df1, df2, default_groupby_columns, excluded_groupby_columns
        )

        groupby_columns = st.multiselect(
            "Select columns to group by:",
            groupby_options,
            default=valid_default_groupby_columns
        )

        selected_metrics = st.multiselect("Select metrics to compare:", numeric_columns)

        if st.button("Compare"):
            group_and_compare(df1, df2, groupby_columns, selected_metrics)

        display_discrepancies()