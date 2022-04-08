# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code first detect the cells in image and then check the accuracy against the ground truth.
"""

import cv2
import json
import csv
import pandas as pd
from sklearn import preprocessing


# replace these paths with yours
images_path = "data/IML_Malaria/"
annotation_path = "annotations.json"
save_annotated_img_path = "data/annotations/"

# %%
with open(annotation_path) as annotation_path:
    ground_truth = json.load(annotation_path)

# %%

le = preprocessing.LabelEncoder()
le.fit(["red blood cell", "gametocyte", "ring", "schizont", "trophozoite", "difficult"])

with open('celltype.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # iterate through all images and find TF and FP.
    for single_image_ground_truth in ground_truth:

        image_name = single_image_ground_truth['image_name']
        objects = single_image_ground_truth['objects']
        image = cv2.imread(images_path + image_name)

        for idx, bbox in enumerate(objects):
            # cell_type = bbox['type']
            x = int(bbox['bbox']['x'])
            y = int(bbox['bbox']['y'])
            h = int(bbox['bbox']['h'])
            w = int(bbox['bbox']['w'])
            crop = image[y:y+h, x:x+w]
            # cv2.imwrite(images_path.replace("IML_Malaria", "cells")+image_name.replace('.', f'_{idx:03d}.'), crop)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
            label = le.transform([bbox['type']])

            if label == 2:
                label = 0
            elif label == 1:
                label = 1
            elif label == 5:
                label = 4
            elif label == 3:
                label = 2
            elif label == 4:
                label = 3
            else:
                label = 5
            writer.writerow({'image_name': image_name.replace('.', f'_{idx:03d}.'), 'label': label})

    cv2.imwrite(save_annotated_img_path + image_name, image)

df = pd.read_csv('celltype.csv')
# One hot encoding for categorical data
# encoding = pd.get_dummies(df[['label']], prefix=None)
# df = df.drop(['label'], axis=1)
# df = pd.concat([df, encoding], axis=1)
# df.to_csv('celltype.csv', index=False)
# print(encoding)
