import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_excel_file(file_path, sheet_name=1):
    # Load the Excel file from the specified sheet without headers
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Find indices where 'Element' is found in column 0 to determine the start of data blocks
    data_blocks_indices = data.index[data[0] == 'Element'].tolist()

    # Define the expected columns for processed data
    expected_headers = ['Element', 'Signal', 'Type', 'Line', 'Apparent Concentration',
                        'Intensity Correction', 'k Ratio', 'Wt%', 'Wt% Sigma', 'Atomic %',
                        'Oxide', 'Oxide %', 'Oxide % Sigma', 'Number of Ions', 'Type of Ion',
                        'Standard Name', 'Factory Standard', 'Standardization Date', 'Beam Current (nA)']

    processed_data = []

    # Iterate through each data block
    for i, start in enumerate(data_blocks_indices):
        # The label for the block is in the row just before the 'Element' row
        label = data.iloc[start - 1, 0]

        # Identify the end of the block by finding the next 'Total' row after 'Element'
        total_index = data.index[(data[0] == 'Total') & (data.index > start)].min()
        end = total_index  # Set end to the row before 'Total'

        # Extract data block for current sample
        block = data.iloc[start + 1:end]  # Extract rows between 'Element' and 'Total'
        headers = ['Element'] + data.iloc[start, 1:].dropna().values.tolist()  # Get headers from the 'Element' row

        # Clean and standardize the column headers (for cases like the truncated or merged headers in the image)
        headers = [str(header).strip() for header in headers]  # Remove leading/trailing whitespace
        headers = [header.replace('\n', ' ').replace('\r', '') for header in headers]  # Clean any special characters

        # Align headers with expected headers by adding missing ones
        if len(headers) < len(expected_headers):
            headers += [None] * (len(expected_headers) - len(headers))

        # Limit block's columns to match available headers
        block.columns = headers[:block.shape[1]]

        # Prepare a row dictionary to append to the DataFrame
        row_dict = {'Sample Name': label}

        for _, row in block.iterrows():
            element = row['Element']

            for col in row.index[1:]:  # Skip the 'Element' column for key names
                # Only proceed if the column exists in the current row and value is not NaN
                if col and pd.notna(row.get(col, None)):  # Use .get() to handle missing columns gracefully
                    col_name = f"{col}_{element}"  # Use header from 'Element' row for column names
                    row_dict[col_name] = row[col]

        # Capture 'Total_wt%' and 'Total_Oxide %' from the 'Total' row if they exist
        total_row = data.iloc[total_index, :]  # Correctly access the 'Total' row
        row_dict['Total_wt%'] = total_row.get(headers.index('Wt%'), None) if 'Wt%' in headers else None
        row_dict['Total_Oxide%'] = total_row.get(headers.index('Oxide %'), None) if 'Oxide %' in headers else None

        processed_data.append(row_dict)

    return pd.DataFrame(processed_data)



def sort_columns(df2, norm = False):

    df=df2.copy()
    # Start with fixed priority columns
    priority_cols = ['Sample Name', 'Total_wt%', 'Total_Oxide%'] #, 'Cation_Sum']

    # Define exact matches for metrics and their intended order
    col_order = ['Oxide %_', 'Oxide % Sigma_', 'norm_Oxide %_', 'Wt%_', 'Wt% Sigma_']


    element_to_oxide = {
        'Si': 'SiO2', 'Ti': 'TiO2', 'Al': 'Al2O3', 'Ca': 'CaO',
        'Cr': 'Cr2O3', 'Fe': 'FeO', 'Ni': 'NiO', 'Na': 'Na2O',
        'Mg': 'MgO','Mn': 'MnO',
        'K': 'K2O', 'P': 'P2O5', 'S': 'SO3'
    }

    new_columns = {}
    for col in df.columns:
        if 'Oxide' in col:
            for element, oxide in element_to_oxide.items():
                if col.endswith(f'_{element}'):
                    new_col_name = col.replace(f'_{element}', f'_{oxide}')
                    new_columns[col] = new_col_name

    # Apply the renaming
    df.rename(columns=new_columns, inplace=True)


    # Find the columns that are going to be involved in the normalization
    desired_string = 'Oxide %_'
    matching_columns = [col for col in df.columns if desired_string in col]
    matching_column_indices = [df.columns.get_loc(col) for col in matching_columns]

    # Convert SO3 to SO2
    # if 'Oxide %_S' in list(df.keys()):
    #     df = df.rename(columns={'Oxide %_S': 'Oxide %_SO3'})
    #     df['Oxide %_SO'] = df['Oxide %_SO2']*(32.065+2*15.999)/(32.065+3*15.999)

    # normalization
    df_norm = df.iloc[:, matching_column_indices]
    df_norm = df_norm.map(lambda x: 0 if x < 0 else x)
    sum = df_norm.sum(axis = 1)
    df_norm = 100*df_norm.div(sum, axis = 0)

    df_norm = df_norm.add_suffix('_norm')
    df_norm = df_norm.add_prefix('norm_')
    df_norm = df_norm.fillna(0.0)

    df = pd.concat([df, df_norm], axis=1)

    # if norm is true, switch normalized and un-normalized analyses
    if norm is True:
        elements = set([col.split('_')[-1].split('_norm')[0] for col in df.columns if 'Oxide %_' in col])

        # Swap the data between corresponding columns
        for element in elements:
            if element != 'norm':
                oxide_col = f'Oxide %_{element}'
                norm_oxide_col = f'norm_Oxide %_{element}_norm'

                # Swap the data using a temporary variable
                temp = df[oxide_col].copy()
                df[oxide_col] = df[norm_oxide_col]
                df[norm_oxide_col] = temp


    # Collect and sort metric-specific columns using more accurate matching
    metric_cols = []
    for metric in col_order:
        # This loop will check if the columns end with the metric name preceded by an underscore
        # to ensure we match 'Oxide %_Mg' but not 'Something Oxide %_Mg'
        metric_cols.extend(sorted([col for col in df.columns if col.startswith(metric)]))


    # Collect all other columns not in priority or metric-specific lists
    other_cols = [col for col in df.columns if col not in priority_cols and col not in metric_cols]

    # Move 'Apparent Concentration' columns to the end
    apparent_concentration_cols = [col for col in other_cols if 'Apparent Concentration' in col]
    other_cols = [col for col in other_cols if col not in apparent_concentration_cols]

    # Combine all columns in the desired order
    final_cols = priority_cols + metric_cols + sorted(other_cols) + sorted(apparent_concentration_cols)

    # Debugging: Print out final column order
    # Move 'Apparent Concentration' columns to the end
    for bad_names in ['Apparent Concentration', 'Factory Standard', 'Line', 'Number of Ions']:

        apparent_concentration_cols = [col for col in df.columns if bad_names in col]
        final_cols = [col for col in final_cols if col not in apparent_concentration_cols] + apparent_concentration_cols

    df_out=df[final_cols]

    # replace column headers to make compatible with MinML

    df_out.rename(columns=lambda x: x.replace('norm_Oxide %_', ''), inplace=True)
    if norm is True:
        df_out.rename(columns=lambda x: x.replace('_norm','_unnorm'), inplace=True)

    df_out.rename(columns=lambda x: x.replace('Oxide %_', ''), inplace=True)
    df_out.rename(columns=lambda x: x.replace('Number of Ions', '#_ions_'), inplace=True)
    df_out.rename(columns=lambda x: x.replace('FeO', 'FeOt'), inplace=True)

    # Reorder DataFrame
    return df_out



# do MinML classification - assign cation totals and split into separate sheets in output excel.
def minClass(df):
    import mineralML as mm
    # prep dataframe for MinML analysis
    Oxides = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    for o in Oxides:
        if o not in df.keys():
            df.loc[:,o] = np.zeros(len(df))

    df2=df[Oxides]
    df2=df2.fillna(0)

    df2['Mineral']='Olivine'
    df2['entity_id']=df.index

    df_pred_nn, probability_matrix = mm.predict_class_prob_nn(df2)

    # Extract key columns into main results dataframe
    # df['Predict_Mineral'] = df_pred_nn['Predict_Mineral']
    # df['Predict_Probabilty'] = df_pred_nn['Predict_Probability']
    columns = df.columns.tolist()
    if 'Predict_Mineral' in columns:
        columns.remove('Predict_Mineral')
    if 'Predict_Probability' in columns:
        columns.remove('Predict_Probability')
    columns.insert(1, 'Predict_Mineral')
    columns.insert(2, 'Predict_Probability')

    df.insert(1, 'Predict_Mineral', df2['Predict_Mineral'])
    df.insert(2, 'Predict_Probability', df2['Predict_Probability'])
    columns = df.columns.tolist()
    if 'Mineral' in columns:
        columns.remove('Mineral')


    df = df.reindex(columns = columns)

    # Oxygen_numbers = {'Amphibole': 22, 'Apatite': 4, 'Biotite': 10, 'Clinopyroxene': 6,
    #                   'Garnet': 12, 'Ilmenite': 3, 'K-Feldspar': 8, 'Magnetite': 4,
    #                   'Muscovite': 10, 'Olivine': 4, 'Orthopyroxene': 6, 'Plagioclase': 8,
    #                   'Quartz': 2, 'Rutile': 2, 'Spinel': 4, 'Tourmaline': 27, 'Zircon': 4}

    # for i in df['Predict_Mineral'].unique():
    #     df.loc[df['Predict_Mineral'] == i, 'Cation_Sum'] = Oxygen_numbers[i]*df.loc[df['Predict_Mineral'] == i, 'Cation_Sum'].astype(float)/oxygen_num

    return df



## Making nice things for the standards to paste into

ideal_cols_standards_silicates=['AnalysisDate', 'PersonName', 'SampleID', 'Total_Oxide%',
 'MgO', 'SiO2', 'TiO2', 'Al2O3', 'CaO', 'MnO', 'P2O5', 'Na2O', 'K2O','FeOt', 'NiO', 'SO3', 'Cr2O3',
  'MgO_norm', 'SiO2_norm', 'TiO2_norm', 'Al2O3_norm', 'CaO_norm', 'MnO_norm', 'P2O5_norm', 'Na2O_norm', 'K2O_norm','FeOt_norm', 'NiO_norm', 'SO3_norm', 'Cr2O3_norm',
                       'Oxide % Sigma_MgO', 'Oxide % Sigma_SiO2', 'Oxide % Sigma_TiO2', 'Oxide % Sigma_Al2O3', 'Oxide % Sigma_CaO', 'Oxide % Sigma_MnO', 'Oxide % Sigma_P2O5',
                     'Oxide % Sigma_Na2O', 'Oxide % Sigma_K2O','Oxide % Sigma_FeOt', 'Oxide % Sigma_NiO', 'Oxide % Sigma_SO3', 'Oxide % Sigma_Cr2O3']

def extract_silicate_standard_data(df_final, PersonName, StdName, AnalysisDate, StdString):
    # Set the required fields
    df_final['PersonName'] = PersonName
    df_final['StdName'] = StdName
    df_final['AnalysisDate'] = AnalysisDate

    # Create a new DataFrame with all ideal columns, initializing missing columns with empty values
    df_standard = pd.DataFrame(columns=ideal_cols_standards_silicates)

    # Fill in values from df_final where the columns match
    for col in df_final.columns:
        if col in df_standard.columns:
            df_standard[col] = df_final[col]

    # Display the new DataFrame
    df_filtered2=df_standard.loc[df_standard['SampleID'].str.contains(StdString)]

    display(df_filtered2.head())

    return df_filtered2


# def extract_silicate_standard_data(df_final, PersonName, StdName, AnalysisDate, StdString):
#
#     df_final['PersonName']=PersonName
#     df_final['StdName']=StdName
#     df_final['AnalysisDate']=AnalysisDate
#
#     # Ensure the columns you want are in the DataFrame
#     filtered_columns = [col for col in ideal_cols_standards_silicates if col in df_final.columns]
#
#     # Create the new DataFrame
#     df_filtered = df_final[filtered_columns]
#
#     # Display the new DataFrame
#     df_filtered2=df_filtered.loc[df_filtered['SampleID'].str.contains(StdString)]
#
#     print(df_filtered2.head())
#
#
#
#     return df_filtered2


