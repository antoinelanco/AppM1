import collections
import pickle

import model
import opendata
import numpy as np
import torch as th
import torch.nn as nn
import torch.autograd as ag

def unpickle(file):
    return pickle.load(open(file, 'rb'), encoding='bytes')

class Perceptron:

    def __init__(self, labels):
        self.labels = labels
        self.weights = collections.defaultdict(lambda: collections.defaultdict(float))
        self.n_updates = 0

    def predict(self, features):

        res = []
        for l in self.labels:
            tmp = 0
            for f in features :
                tmp += self.weights[f][l]
            res.append(tmp)

        return self.labels[res.index(max(res))]

    def score(self, features, labels):
        res = 0
        taille = len(features)
        i = 1
        for f,l in zip(features,labels):
                res += 0 if l == self.predict(f) else 1
                print("Score : "+ str((i/taille)*100) +"%")
                i+=1
        return res/taille


    def update(self, features, labels, alpha=1):
        self.n_updates += 1
        taille = len(features)
        i = 1
        for f,l in zip(features,labels):
            print("Update : "+ str((i/taille)*100) +"%")
            i+=1
            if self.predict(f) == l : continue
            for ff in f:
                self.weights[ff][l] = self.weights[ff][l] + alpha * (1) * ff


    def __getstate__(self):

        return {"weights": {k: v for k, v in self.weights.items()}}

    def __setstate__(self, data):
        self.weights = collections.defaultdict(lambda: collections.defaultdict(float), data["weights"])
        self._accum = None
        self._last_update = None

    def __str__(self):
        return "Nb updates : "+str(self.n_updates)+"\nWeights : "+str(self.weights)



open = unpickle("cifar-10-batches-py/data_batch_1")

filenames = open[b'filenames']
data = open[b'data']
labels = open[b'labels']
batch_label = open[b'batch_label']

print(data.shape)
print(type(labels), len(labels))
batched_tensor_data = opendata.make_tensors_data(data, 50)
batched_tensor_label = opendata.make_tensors_label(labels, 50)

print(len(batched_tensor_data))
# 1er batch
# print(batched_tensor_data[0])
# 1er batch, 1ere image
# print(batched_tensor_data[0][0])

#a = Perceptron(labels)
#a.update(data[:10],labels[:10])
#print ("Loss rate : ", str(a.score(data[:10],labels[:10])*100)+"%")

model = model.MyModel(3072, 10)
learning_rate = 1e-1
loss_fn = nn.NLLLoss()

if th.cuda.is_available():
    model.cuda()
    loss_fn.cuda()

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)


EPOCH = 10

# Boucle d'apprentissage
for i in range(EPOCH):    
    model.train()
    total_loss = 0

    for x, y in zip(batched_tensor_data, batched_tensor_label):
        model.zero_grad()
        x = ag.Variable(x)
        y = ag.Variable(y)

        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    print("Epoch %s, loss = %s" % (i, total_loss))
