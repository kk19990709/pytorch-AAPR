'''
Author: your name
Date: 2021-01-29 09:49:56
LastEditTime: 2021-02-05 11:05:39
LastEditors: your name
Description: In User Settings Edit
FilePath: /pytorch-AAPR/prepropose.py
'''
# %%
import re
from src.utils import *
import json
from collections import Counter
import pandas as pd

#%%
with open('./data/data1', 'r', encoding='utf-8') as f:
    data = json.load(f)
data = pd.json_normalize(data.values(), max_level=0)
with open('./data/data2', 'r', encoding='utf-8') as f:
    temp = json.load(f)
temp = pd.json_normalize(temp.values(), max_level=0)
data = pd.concat([data, temp])
with open('./data/data3', 'r', encoding='utf-8') as f:
    temp = json.load(f)
temp = pd.json_normalize(temp.values(), max_level=0)
data = pd.concat([data, temp])
with open('./data/data4', 'r', encoding='utf-8') as f:
    temp = json.load(f)
temp = pd.json_normalize(temp.values(), max_level=0)
data = pd.concat([data, temp])

#%%
from tqdm import tqdm
counter = Counter([])
tex_data = list(data['tex_data'].copy(deep=True).values)
for i, tex in enumerate(tqdm(tex_data)):
    items = resolve(tex)
    for item in items:
        counter[item] += 1

# %%
print(len(counter))
sum = 0
sum1 = 0
for k, v in counter.items():
    sum += v
    if v > 1:
        sum1 += v
print(sum, sum1)

# %%
sum = 0
for k in counter.most_common(15):
    print(k)
print(sum)

# %%

# %%
