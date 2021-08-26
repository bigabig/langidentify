from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import matplotlib.pylab as plt
import seaborn as sns


langs = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
lang2label = {'deu': 0, 'eng': 1, 'fra': 2,
              'ita': 3, 'por': 4, 'spa': 5}

data_train = pd.read_csv('data/train_sentences.csv', encoding='utf-8')
data_val = pd.read_csv('data/val_sentences.csv', encoding='utf-8')
data_test = pd.read_csv('data/test_sentences.csv', encoding='utf-8')


def visualize_feature_overlaps(lang2feat):
    langs = list(lang2feat.keys())
    feats = [lang2feat[l] for l in langs]
    result = [[len(featA.intersection(featB)) for featB in feats] for featA in feats]
    df = pd.DataFrame(data=result, columns=langs, index=langs)

    ax = sns.heatmap(df, annot=True, linewidth=0.5, fmt="d")
    ax.xaxis.set_ticks_position('top')
    plt.show()


def calc_trigrams(dataset, n=200):
    # init count vectorizer
    vectorizer = CountVectorizer(analyzer='char',
                                 ngram_range=(3, 3),
                                 max_features=n)

    # apply vectorizer on dataset
    vectorizer.fit_transform(dataset)

    # extract features
    features = vectorizer.get_feature_names()

    return features


# calculate common trigrams for each language
features = set()
lang2features = {}
for lang in langs:
    data = data_train[data_train['lang'] == lang]['text']
    trigrams = calc_trigrams(data)
    features.update(trigrams)
    lang2features[lang] = set(trigrams)

# compute vocabulary dict with key = trigram, value = id
vocabulary = {trigram: idx for idx, trigram in enumerate(features)}

# init a count vectorizer that counts our provided trigrams (the trigrams in the vocabulary)
vectorizer = CountVectorizer(analyzer='char',
                             ngram_range=(3, 3),
                             vocabulary=vocabulary)

# apply the vectorizer on the dataset to count the features
train_features = vectorizer.fit_transform(data_train['text'])

# scale the features so that values are in range 0 - 1
scaler = MinMaxScaler()
scaler.fit(train_features.toarray())
train_features = scaler.transform(train_features.toarray())
train_labels = [lang2label[lang] for lang in list(data_train['lang'])]

# repeat for validation and test set
val_features = vectorizer.fit_transform(data_val['text'])
val_features = scaler.transform(val_features.toarray())
val_labels = [lang2label[lang] for lang in list(data_val['lang'])]

test_features = vectorizer.fit_transform(data_test['text'])
test_features = scaler.transform(test_features.toarray())
test_labels = [lang2label[lang] for lang in list(data_test['lang'])]

# save the preprocessed data
with open('data/processed.pkl', 'wb') as f:
    pickle.dump({'train': [train_features, train_labels],
                 'val': [val_features, val_labels],
                 'test': [test_features, test_labels]
                 }, f)

# save the vectorizer for reuse (e.g. doing predictions)
with open('checkpoints/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# save the scaler for reuse (e.g. doing predictions)
with open('checkpoints/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# visualize features
visualize_feature_overlaps(lang2features)
