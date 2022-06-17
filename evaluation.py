# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:18:14 2022

@author: Jerry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score



# In[]

def calculate_WP(y_true, y_pred): # y_pred = y_CoAnet
    
    weighted_percision = precision_score(y_true, y_pred, average='weighted')
    
    return weighted_percision



# In[]

y_df = pd.read_csv('data/test.txt', sep = ' ', header = None, names = ['image_filename', 'label'])
y_true = y_df['label']
class_list = list(np.unique(y_true))

test_num_dict = {}

for index in y_df.index:
    temp_label = y_df.iloc[index, -1]
    
    if temp_label in test_num_dict:
        test_num_dict[temp_label] += 1
    else:
        test_num_dict[temp_label] = 1


label = list(test_num_dict.keys())
label_count = list(test_num_dict.values())

fig, ax = plt.subplots(figsize = (16, 12))
plt.bar(label, label_count)

for p in ax.patches:
   ax.annotate(p.get_height(), (p.get_x() + p.get_width()/4, p.get_height() + 0.02), fontsize = 12)

plt.title('Counts of 14 labels in testing dataset, total 4,028 images', fontsize = 16)
plt.xlabel('label', fontsize = 16)
plt.ylabel('count', fontsize = 16)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('test_count_plot.png')
plt.close()

total = 0
for i_label_count in label_count:
    total += i_label_count

# In[]

y_CoAnet = pd.read_csv('submission_CoAnet50.csv')['label']
y_CoAnet_bal5000 = pd.read_csv('submission_CoAnet_bal5000.csv')['label']
y_CoAnet_aug3000 = pd.read_csv('submission_CoAnet50_aug3000.csv')['label']
y_resnet50 = pd.read_csv('submission_resnet50.csv')['label']

# In[]

WP_CoAnet = calculate_WP(y_true, y_CoAnet)
WP_CoAnet_bal5000 = calculate_WP(y_true, y_CoAnet_bal5000)
WP_CoAnet_aug3000 = calculate_WP(y_true, y_CoAnet_aug3000)
WP_resnet50 = calculate_WP(y_true, y_resnet50)

F1_list_CoAnet = f1_score(y_true, y_CoAnet, average = None)
print(min(F1_list_CoAnet))
F1_list_CoAnet_bal5000 = f1_score(y_true, y_CoAnet_bal5000, average = None)
print(min(F1_list_CoAnet_bal5000))
F1_list_CoAnet_aug3000 = f1_score(y_true, y_CoAnet_aug3000, average = None)
print(min(F1_list_CoAnet_aug3000))
F1_list_resnet50 = f1_score(y_true, y_resnet50, average = None)

# In[] plot WP bar

model_list = ['CoANet', 'CoANet balance 5000', 'CoANet aug 3000', 'ResNet50']
WP_list = [WP_CoAnet, WP_CoAnet_bal5000, WP_CoAnet_aug3000, WP_resnet50]

# plt.bar(model_list, WP_list, color=['red', 'green', 'blue', 'orange'])
# plt.ylim([0.5, 1])
# plt.xticks(rotation = 10)
# plt.title('Weighted precision of models')
# plt.savefig('Weighted precision of models.png')
# plt.close()

df = pd.DataFrame({'model':model_list, 'WP':WP_list})
ax = sns.barplot(x = 'model', y = 'WP', data = df)
ax.set_ylim([0.5, 1])
ax.set_title('Weighted precision of models')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 10)

for index, row in df.iterrows():
    ax.text(row.name, row.WP, round(row.WP, 4), color='black', ha="center")

plt.savefig('Weighted precision of models.png')
plt.close()

# In[] plot F1-score line

plt.figure(figsize=(12, 8))

plt.plot(class_list, F1_list_CoAnet, label = 'CoANet')
plt.plot(class_list, F1_list_CoAnet_bal5000, label = 'CoANet balance 5000')
plt.plot(class_list, F1_list_CoAnet_aug3000, label = 'CoANet aug 3000')
plt.plot(class_list, F1_list_resnet50, label = 'ResNet50')
plt.legend(fontsize = 12)
plt.xticks(rotation = 30)

plt.xlabel('Labels', fontsize = 12)
plt.ylabel('F1-scores', fontsize = 12)
plt.title('F1-scores of models', fontsize = 12)

plt.savefig('F1-scores of models.png')
plt.close()














