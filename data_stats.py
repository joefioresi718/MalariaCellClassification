import pandas as pd


df = pd.read_csv('celltype.csv')

print(df.drop(['image_name', 'cell_num'], axis=1).describe())

for columns in list(df.drop(['image_name', 'cell_num'], axis=1)):
    print(pd.value_counts(df[columns]))

