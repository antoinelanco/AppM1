import collections
import pickle

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
a = Perceptron(labels)
a.update(data[:10],labels[:10])
print ("Loss rate : "str(a.score(data[:10],labels[:10])*100)+"%")
