import streamlit as st
import pandas as pd

# Function to read files
def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

# Streamlit page configuration

st.title("QA Tool") # page title
st.write("The QA Tool compares data between two files, highlighting discrepancies. Upload CSV or Excel files, match column names, and group by selected columns. It provides clear, visual feedback to ensure data consistency and accuracy.")
st.sidebar.image("./images/hn-logo.png", output_format="PNG", use_column_width="always") # logo

# Upload the files
st.write("#### Upload Files:")
file_1 = st.file_uploader("Choose the first file", type=["csv", "xlsx"])
file_2 = st.file_uploader("Choose the second file", type=["csv", "xlsx"])

if file_1 and file_2:
    df1 = read_file(file_1)
    df2 = read_file(file_2)
    
    if df1 is not None and df2 is not None:
        st.markdown("<span style='color:green'><b>Files uploaded successfully</b></span>", unsafe_allow_html=True)
        
        # columns renaming and matching
        columns_1 = list(df1.columns)
        columns_2 = list(df2.columns)
        
        st.write("#### Fix Column Names:") 
        st.write("This tool will only compare the column names that match. If needed, you can manually edit them.")
        st.write("##### Rename columns in File 1 to match File 2:")
        new_columns_1 = {}
        for col in columns_1: # rename file 1 columns 
            new_col = st.text_input(f"Rename '{col}' in File 1", col)
            new_columns_1[col] = new_col
        
        st.write("##### Rename columns in File 2 to match File 1:")
        new_columns_2 = {}
        for col in columns_2: # rename file 2 columns 
            new_col = st.text_input(f"Rename '{col}' in File 2", col)
            new_columns_2[col] = new_col

        df1.rename(columns=new_columns_1, inplace=True)
        df2.rename(columns=new_columns_2, inplace=True)
        
        # get matching columns
        common_columns = list(set(df1.columns) & set(df2.columns))
        
        if common_columns:
            groupby_columns = st.multiselect("Select columns to group by", common_columns)
                
            # button to run results
            if st.button("Compare"):
                if groupby_columns:
                    # Convert groupby columns to the same data type
                    for col in groupby_columns:
                        if df1[col].dtype != df2[col].dtype:
                            try:
                                df1[col] = pd.to_datetime(df1[col])
                                df2[col] = pd.to_datetime(df2[col])
                            except:
                                df1[col] = df1[col].astype(str)
                                df2[col] = df2[col].astype(str)
                    
                    # group by and merge selected columns
                    df1_grouped = df1.groupby(groupby_columns).sum().reset_index()
                    df2_grouped = df2.groupby(groupby_columns).sum().reset_index()
                    merged_df = pd.merge(df1_grouped, df2_grouped, on=groupby_columns, suffixes=(' F1', ' F2'))
                    
                    results = merged_df[groupby_columns].copy()
                    discrepancies_found = False
                    discrepancy_rows = pd.DataFrame()

                    for col in common_columns:
                        if col not in groupby_columns:
                            col_A = f'{col} F1'
                            col_B = f'{col} F2'
                            if col_A in merged_df.columns and col_B in merged_df.columns:
                                # convert columns to numeric to handle string values -- fix issue
                                merged_df[col_A] = pd.to_numeric(merged_df[col_A], errors='coerce')
                                merged_df[col_B] = pd.to_numeric(merged_df[col_B], errors='coerce')

                                # round numbers -- fix issue
                                merged_df[col_A] = merged_df[col_A].round()
                                merged_df[col_B] = merged_df[col_B].round()

                                # include the merged volume columns
                                results[col_A] = merged_df[col_A]
                                results[col_B] = merged_df[col_B]

                                # add comparison calculations
                                results[f'{col} Difference'] = merged_df[col_A] - merged_df[col_B]
                                results[f'{col} % Difference'] = (results[f'{col} Difference'] / merged_df[col_B]) * 100
                                results[f'{col} % Difference'] = results[f'{col} % Difference'].apply(lambda x: f"{x:.2f}%")

                                # check for discrepancies
                                discrepancies = results[f'{col} Difference'] != 0
                                discrepancy_rows = pd.concat([discrepancy_rows, results[discrepancies]]).drop_duplicates().reset_index(drop=True) # dedup results -- fix issue
                                if discrepancies.any():
                                    discrepancies_found = True

                    st.write(results)
                    
                    # creating messages depending on the results
                    if discrepancies_found:
                        number_of_discrepancies = len(discrepancy_rows.index) # count the number of discrepancies
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

# streamlit run 1_Home.py --server.enableXsrfProtection false
