
### MLB project: data preparation steps by Vishva Natarajan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# if you want to run the code the directory has to be changed to where your adult22.csv is downloaded
df = pd.read_csv("/content/drive/MyDrive/MLB_Project/adult22.csv")

columns_tostudy = [
                 'SEX_A','AGEP_A', 'HISPALLP_A','MARITAL_A', 'EDUCP_A','RATCAT_A','HICOV_A', 'EMPLASTWK_A','URBRRL','REGION',\
                 'PCNTKIDS_A','PHSTAT_A', 'CHLEV_A','HYPEV_A', 'CHDEV_A','ANGEV_A', 'MIEV_A','STREV_A', 'ASEV_A',\
                 'CANEV_A', 'DIBEV_A','COPDEV_A', 'ARTHEV_A','ANXEV_A', 'DEPEV_A','HLTHCOND_A', 'HEIGHTTC_A',\
                 'BMICAT_A', 'WEARGLSS_A','VISIONDF_A', 'DIFF_A','SOCWRKLIM_A', 'CVDDIAG_A','SMKEV_A', 'SMOKELSEV_A',\
                 'DRKADVISE1_A', 'PA18_02R_A','WLKLEIS_A', 'ADVACTIVE_A','SLPHOURS_A', 'MEDITATE_A','YOGA_A',\
                 'AFVET_A', 'LSATIS4_A'
                 ]

df_vars = df[columns_tostudy]
# print(len(df_vars.columns))
# df_vars.head()

print("Shape before dropping NaNs: {}".format(df_vars.shape))
# Here I filter out all NaNs from the data
df_vars = df_vars.dropna(subset=columns_tostudy)
print("Shape after dropping rows with any NaNs: {}".format(df_vars.shape))

######### Life satisfaction
# Keep rows with values <= 4 (in effect, remove rows where LS > 4)
df_vars = df_vars[df_vars["LSATIS4_A"] <= 4]

# MANY MANY categories with the "keep 1-2 but not 7-9" rule
# Basically exclude rows with >2

over2 = ['MARITAL_A','HICOV_A','EMPLASTWK_A','SEX_A','PCNTKIDS_A','PHSTAT_A', 'CHLEV_A', 'HYPEV_A','CHDEV_A','ANGEV_A', 'MIEV_A','STREV_A', 'ASEV_A', 'CANEV_A', 'DIBEV_A', 'COPDEV_A', 'ARTHEV_A',\
        'ANXEV_A', 'DEPEV_A','HLTHCOND_A','WEARGLSS_A','VISIONDF_A', 'DIFF_A','SOCWRKLIM_A', 'CVDDIAG_A',\
        'SMKEV_A', 'SMOKELSEV_A','DRKADVISE1_A','WLKLEIS_A', 'ADVACTIVE_A', 'MEDITATE_A','YOGA_A','AFVET_A']
for column in over2:
    df_vars = df_vars[df_vars[column] < 7]


### Now handle all other categories
df_vars = df_vars[df_vars['HISPALLP_A'] < 97]
df_vars = df_vars[df_vars["EDUCP_A"] < 97]
df_vars = df_vars[df_vars["RATCAT_A"] < 98]
df_vars = df_vars[df_vars["PA18_02R_A"] != 8]
df_vars = df_vars[df_vars['AGEP_A'] < 97]
df_vars = df_vars[df_vars["BMICAT_A"] < 9]

df_vars = df_vars[df_vars["HEIGHTTC_A"] < 96]
df_vars = df_vars[df_vars["SLPHOURS_A"] < 97]

# No filtration: REGION, URBRRL,


# df_vars
df_vars
print("Shape: {}".format(df_vars.shape))
print(len(over2))


# one-hot encoding
columns_toencode = ['MARITAL_A','HICOV_A','EMPLASTWK_A','SEX_A','PCNTKIDS_A','PHSTAT_A', 'CHLEV_A', 'HYPEV_A','CHDEV_A','ANGEV_A', 'MIEV_A','STREV_A', 'ASEV_A', 'CANEV_A', 'DIBEV_A', 'COPDEV_A', 'ARTHEV_A',\
        'ANXEV_A', 'DEPEV_A','HLTHCOND_A','WEARGLSS_A','VISIONDF_A', 'DIFF_A','SOCWRKLIM_A', 'CVDDIAG_A',\
        'SMKEV_A', 'SMOKELSEV_A','DRKADVISE1_A','WLKLEIS_A', 'ADVACTIVE_A', 'MEDITATE_A','YOGA_A','AFVET_A',\
                   'HISPALLP_A',"EDUCP_A","RATCAT_A","PA18_02R_A", 'AGEP_A',"BMICAT_A",  'REGION','URBRRL']

print(len(columns_toencode))
df_vars_encoded = pd.get_dummies(df_vars, columns=columns_toencode,dtype=int, drop_first=False)


csv_string = df_vars_encoded.to_csv(index=False)

file_name = '/content/drive/MyDrive/MLB_Project/NHIS_onehot_data.csv'
with open(file_name, 'w') as file:
    file.write(csv_string)


################ Analysis of the correlation to remove any highly correlated variables with correlation >0.8
#### the code was referenced from the following resource: https://www.geeksforgeeks.org/sort-correlation-matrix-in-python/ 
correlation_matrix = df_vars_encoded.corr()
high_corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False)
high_corr_pairs = high_corr_pairs[(high_corr_pairs > 0.8) & (high_corr_pairs < 1)]
print(len(high_corr_pairs))

sns.histplot(df_vars_encoded['LSATIS4_A'])

from collections import Counter
print(Counter(df_vars_encoded['LSATIS4_A']))


