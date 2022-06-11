import pandas as pd
import torch, torch.nn as nn
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split



class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, inputs, labels, max_length):
        self.inputs = inputs
        self.encodings = encodings
        self.labels = labels
        self.avail_classes = np.unique(labels)

    def __getitem__(self, idx):
        item = {}
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['inputs'] = self.inputs[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def Give(opt, datapath, tokenizer, books):
    
    xtrain, ytrain_labels, xtest, ytest_labels = get_train_test(datapath, books)
    ytrain, labels = pd.factorize(ytrain_labels)
    ytest, _ = pd.factorize(ytest_labels)
    # print("ytrain", ytrain)

    train_encodings = opt.tokenizer(xtrain, return_tensors='pt', truncation=True, padding='max_length', max_length=opt.token_max_length)
    test_encodings = opt.tokenizer(xtest, return_tensors='pt', truncation=True, padding='max_length', max_length=opt.token_max_length)

    train_dataset = IMDbDataset(train_encodings, xtrain, ytrain, opt.token_max_length)
    test_dataset = IMDbDataset(test_encodings, xtest, ytest, opt.token_max_length)

    # print("train_labels", train_dataset.avail_classes)
    # print("test_labels", test_dataset.avail_classes)

    return {'training':train_dataset, 'testing':test_dataset}



def get_train_test(file, books):
    X, y = parse_bible(file, books)
    # print(len(X), len(y))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

    return list(Xtrain), list(ytrain), list(Xtest), list(ytest)


def parse_bible(file, books, debug=False):
    tree = ET.parse(file)
    root = tree.getroot()

    verses = []
    labels = []

    for book in root:
        if (book.tag == 'book' and book.attrib['id'] in books):
            if debug: print(book.tag, book.attrib)
            label = book.attrib['id']
            for chapter in book:
                if (chapter.tag == 'c'):
                    if debug: print('Chapter {}'.format(chapter.attrib['id']))
                if (chapter.tag == 'v'):
                    if debug: print('Verse {} - {}'.format(chapter.attrib['id'], chapter.tail))
                    labels.append(label)
                    verses.append(chapter.tail)
            print("LEN:", len(labels))
            # input()
    return verses, labels