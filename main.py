# %%
# region import
import torch
import json
import pandas as pd
from tqdm import tqdm, trange
from nltk.tokenize import word_tokenize
from torch.nn import init, CrossEntropyLoss
from torch.optim import Adam, AdamW
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import GloVe
from args import opt
from src.utils import tokenizer_author, extract
from model.LSTM import LSTM
from model.GRU import GRU
from model.AGRU import AGRU
from model.SharedGRU import SharedGRU
from model.SharedAGRU import SharedAGRU
from model.CNN import CNN
from EarlyStopping import EarlyStopping
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
# region Data set & Preprocessing
if opt.minidata:
    checkpoint = './dataset_mini/data.pkl'
elif opt.largedata:
    checkpoint = './dataset_large/data.pkl'
else:
    checkpoint = './dataset/data.pkl'
# Field
TITLE = Field(tokenize=word_tokenize, lower=True, fix_length=16, init_token='<SOS>')
AUTHO = Field(tokenize=tokenizer_author, lower=True, fix_length=7, init_token='<SOS>')
ABSTR = Field(tokenize=word_tokenize, lower=True, fix_length=160, init_token='<SOS>')
INTRO = Field(tokenize=word_tokenize, lower=True, fix_length=512, init_token='<SOS>')
RELAT = Field(tokenize=word_tokenize, lower=True, fix_length=300, init_token='<SOS>')
METHO = Field(tokenize=word_tokenize, lower=True, fix_length=800, init_token='<SOS>')  # TODO
CONCL = Field(tokenize=word_tokenize, lower=True, fix_length=240, init_token='<SOS>')
VENUE = Field(preprocessing=(lambda x: 0 if x == 'CoRR' or x == 'No' else 1), sequential=False, use_vocab=False)
fields = [('abstr', ABSTR), ('title', TITLE), ('autho', AUTHO), ('categ', None), ('venue', VENUE), ('intro', INTRO), ('relat', RELAT), ('metho', METHO), ('concl', CONCL)]
if opt.makedata:
    with open('./data/data1', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = pd.json_normalize(data.values(), max_level=0)
    if opt.largedata:
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
    print('data processed')
    print('preprocessing')
    examples = [Example.fromlist(list(data.iloc[i]), fields) for i in trange(data.shape[0])]
    torch.save(examples, checkpoint)
    print('preprocess done')
else:
    examples = torch.load(checkpoint)
print('data has just finish been loaded')
# endregion

# %%
# region Dataset & DataLoader
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
AUTHO.build_vocab(train, max_size=1600)

# Iterator
if not opt.notrain:
    train_iter, valid_iter = BucketIterator.splits((train, valid), batch_size=opt.batch_size, sort=False)
test_iter = BucketIterator(test, batch_size=opt.batch_size, sort=False, train=False, shuffle=False)

# endregion

# %%
# region Define the model
# if opt.notrain:
#     model = torch.load(opt.weight_datapath + "model.pt")
#     model.state_dict = torch.load(opt.weight_datapath + './state.pt')
if not opt.notrain:
    if opt.model == 'LSTM':
        model = LSTM().to(device)
    elif opt.model == 'GRU':
        model = GRU().to(device)
    elif opt.model == 'AGRU':
        model = AGRU().to(device)
    elif opt.model == 'SharedGRU':
        model = SharedGRU().to(device)
    elif opt.model == 'SharedAGRU':
        model = SharedAGRU().to(device)
    elif opt.model == 'CNN':
        model = CNN().to(device)
    model.text_embedding_layer.weight.data.copy_(TITLE.vocab.vectors).to(device)
    for para in model.text_embedding_layer.parameters():
        para.requires_grad = False
    # FIXME speedup
    print('define model done')
Watcher = EarlyStopping(model, 5, max_steps=opt.num_epochs)
# endregion

# %%
# region Criterion & Optimizer
if not opt.notrain:
    criterion = CrossEntropyLoss()
    if opt.optimizer == 'Adam':
        optimizer = Adam(model.parameters())
    if opt.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters())
    # 使用1Cycle实现加速 FIXME
    torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.15, epochs=opt.num_epochs, steps_per_epoch=len(train_iter))
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
            text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro, 'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl, 'autho': batch.autho}
            text = {key: item.to(device) for key, item in text.items()}
            venue = torch.as_tensor(batch.venue, dtype=torch.long).to(device)
            # Forward pass
            if opt.model == 'AGRU' or opt.model == 'SharedAGRU':
                attn_inputs = torch.ones(opt.batch_size, dtype=torch.long).to(device) * TITLE.vocab.stoi['<SOS>']
                outputs = model(text, attn_inputs)
            else:
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
            text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro, 'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl, 'autho': batch.autho}
            text = {key: item.to(device) for key, item in text.items()}
            venue = torch.as_tensor(batch.venue, dtype=torch.long).to(device)
            # Forward pass
            if opt.model == 'AGRU' or opt.model == 'SharedAGRU':
                attn_inputs = torch.ones(opt.batch_size, dtype=torch.long).to(device) * TITLE.vocab.stoi['<SOS>']
                outputs = model(text, attn_inputs)
            else:
                outputs = model(text)
            loss = criterion(outputs, venue)
            # Display
            history_valid.append(loss.item())
        valid_loss = sum(history_valid) / len(history_valid)
        # print(history_valid)

        correct = 0.0
        total = 0.0
        for i, batch in enumerate(valid_iter):
            text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro, 'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl, 'autho': batch.autho}
            text = {key: item.to(device) for key, item in text.items()}
            venue = torch.as_tensor(batch.venue, dtype=torch.long).to(device)
            if opt.model == 'AGRU' or opt.model == 'SharedAGRU':
                attn_inputs = torch.ones(opt.batch_size, dtype=torch.long).to(device) * TITLE.vocab.stoi['<SOS>']
                outputs = model(text, attn_inputs)
            else:
                outputs = model(text)
            outputs = torch.max(outputs, dim=1).indices
            correct += torch.sum(venue == outputs)
            total += opt.batch_size
        accuracy = correct / total
        if Watcher.need_to_stop(model, accuracy):
            break

        # train_writer.add_scalar('loss', train_loss, epoch)
        # valid_writer.add_scalar('loss', valid_loss, epoch)
        print(f'Epoch [{epoch + 1:<3}/ {opt.num_epochs}], Step [{total_step} / {total_step}], TrainLoss: {train_loss:.4f}, ValidLoss: {valid_loss:.4f}, Accuracy: {accuracy:.2%}')
    # torch.save(model, opt.weight_datapath + "model.pt")
    # torch.save(model.state_dict(), opt.weight_datapath + './state.pt')
# endregion

# %%
# region Test the model
model = torch.load(opt.weight_datapath + "model.pt")
model.state_dict = torch.load(opt.weight_datapath + './state.pt')
correct = 0.0
total = 0.0
for i, batch in enumerate(test_iter):
    text = {'abstr': batch.abstr, 'title': batch.title, 'intro': batch.intro, 'relat': batch.relat, 'metho': batch.metho, 'concl': batch.concl, 'autho': batch.autho}
    text = {key: item.to(device) for key, item in text.items()}
    venue = torch.as_tensor(batch.venue, dtype=torch.long).to(device)
    if opt.model == 'AGRU' or opt.model == 'SharedAGRU':
        attn_inputs = torch.ones(opt.batch_size, dtype=torch.long).to(device) * TITLE.vocab.stoi['<SOS>']
        outputs = model(text, attn_inputs)
    else:
        outputs = model(text)
    outputs = torch.max(outputs, dim=1).indices
    correct += torch.sum(venue == outputs)
    total += opt.batch_size
print(f'{correct/total:.2%}')
# endregion
