# %%
# region import
import torch
import json
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm, trange
from nltk.tokenize import word_tokenize
from torch.nn import init, CrossEntropyLoss
from torch.optim import Adam
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import GloVe
from args import opt
from src.utils import tokenizer_author, tokenizer_category, extract
from model.Scorer import Scorer
print('import done')
# endregion

# %%
# region args
if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')
print(torch.cuda.get_device_name(), device)
# endregion
# %%
# region Data set & Data loader
if opt.minidata:
    checkpoint = './dataset_mini/checkpoint.pkl'
else:
    checkpoint = './dataset/checkpoint.pkl'
if opt.makedata:
    with open('./data/data1', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = pd.json_normalize(data.values(), max_level=0)
    if opt.minidata:
        data = data.head(160)
    else:
        data = data.head(opt.datasize)
    name_dict = {'abstract': 'abstr', 'authors': 'autho', 'category': 'categ'}
    data = data.rename(columns=name_dict)
    data['intro'] = None
    data['relat'] = None
    data['metho'] = None
    data['concl'] = None
    tex_data = list(data['tex_data'].copy(deep=True).values)
    for i, tex in enumerate(tqdm(tex_data)):
        intro, relat, metho, concl = extract(tex)
        data['intro'].iloc[i] = intro
        data['relat'].iloc[i] = relat
        data['metho'].iloc[i] = metho
        data['concl'].iloc[i] = concl
    data = data.drop(columns='tex_data')
    torch.save(data, checkpoint)
    print('data has just finish been processed')
else:
    data = torch.load(checkpoint)
    print('data has just finish been loaded')
# endregion

# %%
# region Preprocessing
# Field
TITLE = Field(tokenize=word_tokenize, lower=True)
AUTHO = Field(tokenize=tokenizer_author)
ABSTR = Field(tokenize=word_tokenize, lower=True)
CATEG = Field(tokenize=tokenizer_category)
INTRO = Field(tokenize=word_tokenize, lower=True)
RELAT = Field(tokenize=word_tokenize, lower=True)
METHO = Field(tokenize=word_tokenize, lower=True)
CONCL = Field(tokenize=word_tokenize, lower=True)
VENUE = Field(preprocessing=(lambda x: 0 if x == 'CoRR' or x == 'No' else 1), sequential=False, use_vocab=False)
fields = [('abstr', ABSTR), ('title', TITLE), ('autho', AUTHO),
          ('categ', CATEG), ('venue', VENUE), ('intro', INTRO),
          ('relat', RELAT), ('metho', METHO), ('concl', CONCL)]

# Dataset
examples = [Example.fromlist(list(data.iloc[i]), fields) for i in range(data.shape[0])]
dataset = Dataset(examples, fields)
train, valid, test = dataset.split(split_ratio=[0.6, 0.2, 0.2], stratified=False, strata_field='label')
# vocab
vectors = GloVe(name='6B', dim=300)
source = [getattr(dataset, item) for item in ['title', 'abstr', 'intro', 'relat', 'metho', 'concl']]
TITLE.build_vocab(source, vectors=vectors, max_size=opt.vocab_size)
TITLE.vocab.vectors.unk_init = init.xavier_uniform
ABSTR.vocab = TITLE.vocab
INTRO.vocab = TITLE.vocab
RELAT.vocab = TITLE.vocab
METHO.vocab = TITLE.vocab
CONCL.vocab = TITLE.vocab
AUTHO.build_vocab(train)
CATEG.build_vocab(train)

# Iterator
if not opt.notrain:
    train_iter, valid_iter = BucketIterator.splits((train, valid), batch_size=opt.batch_size, sort=False)
test_iter = BucketIterator(test, batch_size=opt.batch_size, sort=False, train=False, shuffle=False)
print('preprocess done')
# endregion

# %%
# region Define the model
if opt.notrain:
    model = torch.load(opt.weight_datapath+"model.pt")
    model.state_dict = torch.load(opt.weight_datapath+'./state.pt')
else:
    model = Scorer(opt)
    model.text_embedding_layer.weight.data.copy_(TITLE.vocab.vectors)
    # dummy_input = torch.zeros((opt.batch_size, opt.seq_len), dtype=torch.long).to(device)
    # model_writer.add_graph(model, (dummy_input, ))
    print('define model done')
# endregion

# %%
# region Criterion & Optimizer
if not opt.notrain:
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=opt.learning_rate)
    print('criterion & Optimizer done')
# endregion

# %%
# region Train the model
if not opt.notrain:
    total_step = len(train_iter)
    for epoch in range(opt.num_epochs):
        model.train()
        history_train = []
        for batch in tqdm(train_iter):
            text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro,
                    'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl}
            venue = torch.as_tensor(batch.venue, dtype=torch.long)
            # Forward pass
            outputs = model(text)
            loss = criterion(outputs, venue)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Display
            history_train.append(loss.item())
        train_loss = sum(history_train) / len(history_train)

        model.eval()
        history_valid = []
        for i, batch in enumerate(valid_iter):
            text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro,
                    'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl}
            venue = torch.as_tensor(batch.venue, dtype=torch.long)
            # Forward pass
            outputs = model(text)
            loss = criterion(outputs, venue)
            # Display
            history_valid.append(loss.item())
        valid_loss = sum(history_valid) / len(history_valid)
        # train_writer.add_scalar('loss', train_loss, epoch)
        # valid_writer.add_scalar('loss', valid_loss, epoch)
        print(f'Epoch [{epoch + 1:<3}/ {opt.num_epochs}], Step [{total_step} / {total_step}], TrainLoss: {train_loss:.4f}, ValidLoss: {valid_loss:.4f}')
    torch.save(model, opt.weight_datapath+"model.pt")
    torch.save(model.state_dict(), opt.weight_datapath+'./state.pt')
# endregion

# %%
# region Test the model
correct = 0.0
total = 0.0
for i, batch in enumerate(test_iter):
    text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro,
            'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl}
    venue = torch.as_tensor(batch.venue, dtype=torch.long)
    outputs = torch.max(model(text), dim=1).indices
    correct += torch.sum(venue == outputs)
    total += opt.batch_size
print(f'{correct/total:.2f}')
# endregion
