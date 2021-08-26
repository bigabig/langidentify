import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/sentences.csv',
                   sep='\t',
                   encoding='utf8',
                   index_col=0,
                   names=['lang', 'text'])

# print all available languages
# langs = data['lang'].unique()
# print(langs)

# we are only interested in a few
lang_filter = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
data = data[data['lang'].isin(lang_filter)]

# filter by sentence length (20 - 200 tokens)
len_filter = [20 <= len(sent) <= 200 for sent in data['text']]
data = data[len_filter]

# sample data for each language
samplesize = 50000
dataset = pd.DataFrame(columns=['lang', 'text'])
for lang in lang_filter:
    temp = data[data['lang'] == lang].sample(samplesize, random_state=42)
    dataset = dataset.append(temp)

# create train (80%), validation(10%), test (10%) splits
train, test = train_test_split(dataset, test_size=1/10, shuffle=True, random_state=42)
train, val = train_test_split(train, test_size=1/9, shuffle=True, random_state=42)

# save datasets
train.to_csv('data/train_sentences.csv', encoding='utf-8', index=False)
test.to_csv('data/test_sentences.csv', encoding='utf-8', index=False)
val.to_csv('data/val_sentences.csv', encoding='utf-8', index=False)
