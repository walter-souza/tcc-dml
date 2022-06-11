# from google.auth import default
# import gspread
# import pandas as pd
# import torch, torch.nn as nn
# import numpy as np


# class IMDbDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, inputs, labels, max_length):
#         self.inputs = inputs
#         # self.encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=max_length)
#         self.encodings = encodings
#         self.labels = labels
#         self.avail_classes = np.unique(labels)

#     def __getitem__(self, idx):
#         item = {}
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['inputs'] = self.inputs[idx]
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# def Give(opt, datapath, tokenizer):
#     creds, _ = default()
#     gc = gspread.authorize(creds)

#     worksheet = gc.open(datapath).get_worksheet(0)
#     rows = worksheet.get_all_values()
    
#     xtrain, ytrain, xtest, ytest = get_train_test(rows)
#     # print("ytrain", ytrain)
#     train_encodings = opt.tokenizer(xtrain, return_tensors='pt', truncation=True, padding='max_length', max_length=opt.token_max_length)
#     test_encodings = opt.tokenizer(xtest, return_tensors='pt', truncation=True, padding='max_length', max_length=opt.token_max_length)

#     train_dataset = IMDbDataset(train_encodings, xtrain, ytrain, opt.token_max_length)
#     test_dataset = IMDbDataset(test_encodings, xtest, ytest, opt.token_max_length)

#     # print("train_labels", train_dataset.avail_classes)
#     # print("test_labels", test_dataset.avail_classes)

#     return {'training':train_dataset, 'testing':test_dataset}

# def get_train_test(rows):
#     intencoes = rows[0][2:]
#     # print("LABELS:", intencoes, rows[0])
#     df = pd.DataFrame(rows[1:],columns=rows[0])
#     #df = df.drop(['finalizar_pedido'], axis=1)
#     #df = df.drop(['endereco_entrega'], axis=1)
#     #df = df.drop(['consultar_pedido'], axis=1)
#     df = df.drop(['data'], axis = 1)

#     xtrain_global = []
#     ytrain_global = []
#     for intencao in intencoes:
#         lintencao = df[df['conjunto']=='Treino'][intencao].values.tolist()
#         xtrain_global += lintencao
#         ytrain_global += [intencao]*len(lintencao)

#     xtest_global = []
#     ytest_global = []
#     for intencao in intencoes:
#         lintencao = df[df['conjunto']=='Teste'][intencao].values.tolist()
#         xtest_global += lintencao
#         ytest_global += [intencao]*len(lintencao)

#     ytrain, labels = pd.factorize(ytrain_global)
#     ytest, _ = pd.factorize(ytest_global)
#     xtrain = xtrain_global
#     xtest = xtest_global

#     # print("ytrain", ytrain, len(ytrain))

#     indices_train = np.arange(len(ytrain))
#     indices_test = np.arange(len(ytest))
#     np.random.shuffle(indices_train)
#     np.random.shuffle(indices_test)

#     xtrain = np.array(xtrain)[indices_train]
#     ytrain = np.array(ytrain)[indices_train]
#     xtest = np.array(xtest)[indices_test]
#     ytest = np.array(ytest)[indices_test]
    
#     # print("ytrain", ytrain, len(ytrain))

#     return list(xtrain), list(ytrain), list(xtest), list(ytest)