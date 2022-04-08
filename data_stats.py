import pandas as pd
from sklearn.utils import class_weight
import numpy as np


# df = pd.read_csv('files/train.csv')
# df_val = pd.read_csv('files/val.csv')
#
# print(df.drop(['image_name'], axis=1).describe())
#
# for columns in list(df.drop(['image_name'], axis=1)):
#     print(pd.value_counts(df[columns]))
#     print(pd.value_counts(df_val[columns]))

df = pd.read_csv('files/celltype.csv')
print(pd.value_counts(df['label'].to_numpy()))

length = len(df)
print(length)

weights = []
for count in pd.value_counts(df['label'].to_numpy()):
    weights.append((length - count) / length)

print(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df['label'].to_numpy()), y=df['label'].to_numpy()))

print(weights)

