'''
Author: your name
Date: 2021-01-22 09:27:16
LastEditTime: 2021-02-05 11:06:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-AAPR/virs.py
'''
# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from args import opt
checkpoint = './dataset/checkpoint.pkl'
data = torch.load(checkpoint)
print(data.columns)

# %%
data = data[opt.text_section]
descript = data.applymap(lambda x: len(x))
descript

# %%
plt.figure(figsize=(18, 10))
for i, column in enumerate(descript.columns):
    ax = plt.subplot(2, 3, i + 1)
    plt.boxplot(x=descript[column])
plt.tight_layout()

# %%
plt.figure(figsize=(18, 10))
for i, column in enumerate(descript.columns):
    ax = plt.subplot(2, 3, i + 1)
    ax = sns.distplot(descript[column])
plt.tight_layout()

# %%
