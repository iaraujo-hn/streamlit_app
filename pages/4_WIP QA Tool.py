# Streamlit cheats sheet: https://cheat-sheet.streamlit.app/

# st.title()
# st.write()
# st.sidebar.image()
# st.file_uploader()
# st.dataframe()
# st.session_state.somefunction() -- used to store information
# st.button()
# st.markdown()
# st.error()

# Additions to come:
# 1: Add reset button
# 2: Undo button or ensure that, if the automatic matching is not correct, the user has the chance to manually correct. Renaming was not working in previous version
# 3: Show rows that did not match in group by. For example, to find campaign names that might not exist in one of the files
# 4: Ensure that column names contains id must be seen as categorical. What are the consequences of this? Could it cause issues with other metrics?

import streamlit as st
import pandas as pd
from rapidfuzz import process
import spacy

# load spacy model for NLP matching
nlp = spacy.load('en_core_web_sm')

# custom mapping for context-specific matching ######## maybe add the mapping
custom_mapping = {
    'day': 'date',
    'date':'day',
    'date_aired':'date',
    'date':'date_aired',
    'from_date': 'date',
    'date':'from_date',
    'date':'to_date',
    'to_date': 'date',
    'engine': 'site',
    'site':'engine',
    'cost': 'spend',
    'spend':'cost',
    'spend':'spent',
    'spent':'spend'
}

# exceptions dictionary
exceptions = {
    'conversions': ['impressions']
}

# function to read files # note: this fixes an issue where the page would load for a few seconds every time we updated the column names
@st.cache_data
def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

# function to find the best match for column names using fuzzy matching, custom mapping, and exceptions
def find_best_match(input_col, columns_list):
    # check custom mapping
    input_col_lower = input_col.lower()
    if input_col_lower in custom_mapping:
        return custom_mapping[input_col_lower]
    
    # check exceptions
    if input_col_lower in exceptions:
        for exception in exceptions[input_col_lower]:
            if exception in columns_list:
                columns_list.remove(exception)
    
    # if not found in custom mapping or exceptions, try rapidfuzzy matching
    result = process.extractOne(input_col, columns_list, score_cutoff=80)
    if result:
        match = result[0]
        return match
    
    # if fuzzy matching doesn't give a good match, use nlp
    input_doc = nlp(input_col)
    best_match = input_col
    best_similarity = 0
    
    for col in columns_list:
        col_doc = nlp(col)
        similarity = input_doc.similarity(col_doc)
        if similarity > best_similarity:
            best_match = col
    
    return best_match if best_similarity > 0.4 else input_col

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit page configuration #

st.title("QA Tool") # title
st.write("The QA Tool compares data between two files, highlighting discrepancies. Upload CSV or Excel files, match column names, and group by selected columns. It provides clear feedback to ensure data consistency and accuracy.")
st.sidebar.image("./images/hn-logo.png", output_format="PNG", use_column_width="always") # logo in the side bar

# upload the files
st.write("#### Upload Files:")
file_1 = st.file_uploader("Choose the first file", type=["csv", "xlsx"], key="file_1")
file_2 = st.file_uploader("Choose the second file", type=["csv", "xlsx"], key="file_2")

if file_1 and file_2:
    df1 = read_file(file_1)
    df2 = read_file(file_2)
    
    if df1 is not None and df2 is not None:
        st.markdown("<span style='color:green'><b>Files uploaded successfully</b></span>", unsafe_allow_html=True)
        # adding else at the end of the code

        
        # store the original dataframes for resetting 
        ############## ISSUE TO BE FIXED: UNABLE TO RESET PAGE. PREVIOUS FUNCTIONS WERE DELETED. I NEED TO SEARCH FOR A SOLUTON ##############
        # if "original_df1" not in st.session_state:
        #     st.session_state.original_df1 = df1.copy()
        # if "original_df2" not in st.session_state:
        #     st.session_state.original_df2 = df2.copy()
  
        # toggle to view/hide data
        if "show_data" not in st.session_state:
            st.session_state.show_data = False

        if st.button("View Data"): # by default, hide data
            st.session_state.show_data = not st.session_state.show_data
        
        if st.session_state.show_data: # when button is clicked, do the following:
            st.write("### Data from File 1")
            st.write(df1.head())
            st.write("### Data from File 2")
            st.write(df2.head())

#################### COLUMN RENAMING MESSAGE ####################

        # columns renaming and matching
        columns_1 = list(df1.columns)
        columns_2 = list(df2.columns)
        st.markdown("---")
        st.write("### Fix Column Names:") 
        st.write("This tool compares only the matching column names between the two files. It automatically corrects similar column names. \n\nYou can further edit the column names by clicking on 'View Column Names.")
        
        new_columns_2 = {}
        column_replacements = {}

        for col in columns_2:
            matched_col = find_best_match(col, columns_1) # applying matching function to column names
            if matched_col != col:
                column_replacements[col] = matched_col
            new_columns_2[col] = matched_col
        
        df2.rename(columns=new_columns_2, inplace=True)

        # display column replacements message
        if column_replacements:
            st.write("#### Column Name Replacements:")
            for old_col, new_col in column_replacements.items():
                st.markdown(f" Column name '{old_col}' in File 2 was replaced by '{new_col}'")
        
        # view/edit column names part

#################### VIEW AND EDIT COLUMN NAMES ####################
        if "show_edit_columns" not in st.session_state:
            st.session_state.show_edit_columns = False
        
        if st.button("View Column Names"):
            st.session_state.show_edit_columns = not st.session_state.show_edit_columns
        
        if st.session_state.show_edit_columns:
            st.write("##### Rename columns in File 1 to match File 2, if needed:")
            cols1 = st.columns(2)
            for idx, col in enumerate(columns_1):  # rename file 1 columns
                with cols1[idx % 2]:
                    new_col = st.text_input(f"Rename '{col}' in File 1", col, key=f"file_1_{col}_{idx}")
                    df1.rename(columns={col: new_col}, inplace=True)

            st.markdown("---")
            st.write("##### Rename columns in File 2 to match File 1, if needed:")
            cols2 = st.columns(2)
            for idx, col in enumerate(new_columns_2.values()):  # rename file 2 columns
                with cols2[idx % 2]:
                    new_col = st.text_input(f"Rename '{col}' in File 2", col, key=f"file_2_{col}_{idx}")
                    df2.rename(columns={col: new_col}, inplace=True)
            
            # option to undo changes and revert to original data #
            ############## ISSUE TO BE FIXED: UNABLE TO RESET PAGE. PREVIOUS FUNCTIONS WERE DELETED. I NEED TO SEARCH FOR A SOLUTON ##############
            # if st.button("Undo Changes"):
            #     df1 = st.session_state.original_df1.copy()
            #     df2 = st.session_state.original_df2.copy()
            #     st.experimental_rerun()


#################### GROUP BY AND COMPARISON ####################
        common_columns = list(set(df1.columns) & set(df2.columns))
        st.markdown("---")
        if common_columns:
            groupby_columns = st.multiselect("##### Select columns to group by", common_columns, key="groupby_columns")
           
            # button to run results
            if st.button("Compare", key="compare_button"):
                if groupby_columns:
                    # convert groupby columns to the same data type | note: it fixes issues where dates with different formats were throwing an error
                    for col in groupby_columns:
                        if col in df1.columns and col in df2.columns:
                            if df1[col].dtype != df2[col].dtype:
                                try:
                                    df1[col] = pd.to_datetime(df1[col])
                                    df2[col] = pd.to_datetime(df2[col])
                                except:
                                    df1[col] = df1[col].astype(str)
                                    df2[col] = df2[col].astype(str)
                            if '_id' in col.lower():  # ensure ID columns are seeing as categorical
                                df1[col] = df1[col].astype('category')
                                df2[col] = df2[col].astype('category')
                    
                    # add unique suffix to all columns to prevent duplicates
                    df1.columns = [f"{col}_F1" if col not in groupby_columns else col for col in df1.columns]
                    df2.columns = [f"{col}_F2" if col not in groupby_columns else col for col in df2.columns]

                    # group by and merge selected columns
                    df1_grouped = df1.groupby(groupby_columns).sum().reset_index()
                    df2_grouped = df2.groupby(groupby_columns).sum().reset_index()
                    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns)

                    results = merged_df[groupby_columns].copy()
                    discrepancies_found = False
                    discrepancy_rows = pd.DataFrame()

                    for col in common_columns:
                        if col not in groupby_columns and '_id' not in col.lower():  # Skip comparison for _id columns
                            col_A = f"{col}_F1"
                            col_B = f"{col}_F2"
                            if col_A in merged_df.columns and col_B in merged_df.columns:
                                # convert columns to numeric to handle string values | note: fixes issue
                                merged_df[col_A] = pd.to_numeric(merged_df[col_A], errors='coerce')
                                merged_df[col_B] = pd.to_numeric(merged_df[col_B], errors='coerce')

                                # round numbers | note: fixes issue
                                merged_df[col_A] = merged_df[col_A].round()
                                merged_df[col_B] = merged_df[col_B].round()

                                # include the merged volume columns
                                results[col_A] = merged_df[col_A]
                                results[col_B] = merged_df[col_B]

                                # add comparison calculations
                                results[f'{col} Difference'] = merged_df[col_A] - merged_df[col_B]
                                results[f'{col} % Difference'] = (results[f'{col} Difference'] / merged_df[col_B]) * 100
                                results[f'{col} % Difference'] = results[f'{col} % Difference'].apply(lambda x: f"{x:.2f}%")

                                # creating dataframe that will include only values where differences were found

                                ####### NEED TO FIND SOLUTION FOR CODE BELOW AS IT'S DUPLICATING DATA 
                                ####### solution may have been found. need to explore more
                                first_dataframe_retrieved = False

                                discrepancies = results[f'{col} Difference'] != 0
                                if discrepancies.any() and not first_dataframe_retrieved:
                                    discrepancy_rows = results[discrepancies].drop_duplicates().reset_index(drop=True)
                                    first_dataframe_retrieved = True 

                                ####### code commented out below is what I was previously using
                                # discrepancies = results[f'{col} Difference'] != 0
                                # discrepancy_rows = pd.concat([discrepancy_rows, results[discrepancies]]).drop_duplicates().reset_index(drop=True)
                                if discrepancies.any():
                                    discrepancies_found = True
                    st.write('#### Side by Side Comparison')
                    st.write(results)
                    
                    # creating messages depending on the results
                    if discrepancies_found:
                        number_of_discrepancies = len(discrepancy_rows.index)
                        st.markdown(f"<span style='color:red'><b>There are <u>{number_of_discrepancies}</u> rows with discrepancies.</b></span>", unsafe_allow_html=True)

                        st.write("#### Discrepancy Rows:")
                        st.write(discrepancy_rows)
                    else:
                        st.markdown("<span style='color:green'><b>There are no discrepancies between the files.</b></span>", unsafe_allow_html=True)
                else:
                    st.error("Please select at least one column to group by")
        else:
            st.error("The files do not have common columns")
    else:
        st.error("Error reading files")

st.markdown("---")
