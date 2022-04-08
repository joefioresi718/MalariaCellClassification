import pandas as pd
import glob2 as glob
import random
import csv


annotations = pd.read_csv('../celltype.csv')
data = glob.glob('../data/cells/*')
data = sorted([data.split('/')[3] for data in data])
annotations = annotations.sort_values('image_name')
print(data[556])
print(annotations.iloc[556])

train = []
val = []

for i in range(len(data)):
    if random.random() > 0.8:
        val.append({"image_name": data[i], "cell_type": annotations.iloc[i].drop('image_name').to_numpy()[0]})
    else:
        train.append({"image_name": data[i], "cell_type": annotations.iloc[i].drop('image_name').to_numpy()[0]})


keys_t = train[0].keys()
keys_v = val[0].keys()

t_file = open("train.csv", "w")
v_file = open("val.csv", "w")

train_writer = csv.DictWriter(t_file, keys_t)
train_writer.writeheader()
train_writer.writerows(train)
t_file.close()

val_writer = csv.DictWriter(v_file, keys_v)
val_writer.writeheader()
val_writer.writerows(val)
v_file.close()



