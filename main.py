import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LangModel
from text_dataset import TextDataset
import pickle
from enum import Enum

torch.manual_seed(420)
device = "cuda" if torch.cuda.is_available() else "cpu"
langs = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
lang2label = {'deu': 0, 'eng': 1, 'fra': 2,
              'ita': 3, 'por': 4, 'spa': 5}


class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    TEST = 3
    PREDICT = 4


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batchnum, batch in enumerate(dataloader):
        X, y = batch['data'].to(device), batch['label'].to(device)

        pred = model(X)
        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print every 100 batches
        if batchnum % 100 == 0:
            loss, current = loss.item(), batchnum * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate(dataloader, model, loss_function, test=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch['data'].to(device), batch['label'].to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            predictions.extend(pred.argmax(1).tolist())
            labels.extend(y.tolist())

    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"{'Test' if test else 'Validation'} Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return predictions, labels


def predict(dataloader, model):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch['data'].to(device)
            pred = model(X)
            predictions.extend([langs[p] for p in pred.argmax(1).tolist()])
    return predictions


def predict_pipeline(model, data):
    """
    :param model: a trained LangModel
    :param data: either path to *.txt file or list of sentences
    :return: list of predictions e.g. ['deu', 'eng', ...]
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import MinMaxScaler

    # load vectorizer
    with open('checkpoints/vectorizer.pkl', 'rb') as f:
        vectorizer: CountVectorizer = pickle.load(f)

    # load scaler
    with open('checkpoints/scaler.pkl', 'rb') as f:
        scaler: MinMaxScaler = pickle.load(f)

    # load text data
    if not isinstance(data, list):
        with open(data, "r") as f:
            data = f.readlines()
    labels = [0 for i in range(len(data))]

    # preprocess data
    features = vectorizer.fit_transform(data)
    features = scaler.transform(features.toarray())

    # create dataloader
    dataloader = DataLoader(TextDataset(features, labels), batch_size=100)

    # predict
    predictions = predict(dataloader, model)

    # print output
    for text, prediction in zip(data, predictions):
        print(f'Language: {prediction} | Input: {text.strip()}')

    return predictions


def main(mode: Mode = Mode.TRAIN, checkpoint: str = None, data: str = "data/processed.pkl"):
    # hyper parameters
    batch_size = 100

    # init model
    model = LangModel().to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    # init loss
    loss_function = nn.CrossEntropyLoss()

    # load data
    if mode == Mode.PREDICT:
        predict_pipeline(model, data)

    else:
        # load preprocessed data (train, val, test) from file
        with open(data, 'rb') as f:
            data = pickle.load(f)

    if mode == Mode.TRAIN:
        # create dataloaders
        train_dataloader = DataLoader(TextDataset(*data['train']), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(TextDataset(*data['val']), batch_size=batch_size)

        # init optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # training loop
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_function, optimizer)
            validate(val_dataloader, model, loss_function)
        print("Done training!")

        # save model
        torch.save(model.state_dict(), "checkpoints/model.pth")
        print("Saved PyTorch Model State to checkpoints/model.pth")

    elif mode == Mode.EVAL or mode == Mode.TEST:
        from sklearn.metrics import accuracy_score, confusion_matrix
        import matplotlib.pylab as plt
        import seaborn as sns

        dataloader = DataLoader(TextDataset(*data['val' if mode == Mode.EVAL else 'test']), batch_size=batch_size)
        predictions, labels = validate(dataloader, model, loss_function, test=mode == Mode.TEST)

        print(f'Accuracy: {accuracy_score(labels, predictions) * 100:.2f}%')
        matrix = pd.DataFrame(data=confusion_matrix(labels, predictions), columns=langs, index=langs)
        print(matrix)

        sns.set(font_scale=1.2)
        ax = sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt='.5g', cbar=False)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.xlabel('Predicted', fontsize=22)
        plt.ylabel('Actual', fontsize=22)
        plt.show()

        print("Done!")


if __name__ == '__main__':
    # main(mode=Mode.TRAIN, data="data/processed.pkl")
    # main(mode=Mode.EVAL, checkpoint="checkpoints/model.pth")
    main(mode=Mode.TEST, checkpoint="checkpoints/model.pth")
    # main(mode=Mode.PREDICT, checkpoint="checkpoints/model.pth", data="data/input.txt")
