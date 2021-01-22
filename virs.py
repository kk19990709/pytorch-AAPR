# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from args import opt
from copy import deepcopy
checkpoint = './dataset_mini/checkpoint.pkl'
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
