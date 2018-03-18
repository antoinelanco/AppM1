import torch as th
import torch.nn as nn
import torch.autograd as ag

class MyModel(nn.Module):

	def __init__(self, data_size, label_size):
		super(MyModel, self).__init__()
		self.label_size = label_size
		self.linear1 = nn.Linear(data_size, self.label_size)
		self.log_softmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs):
		out = self.linear1(inputs)
		return self.log_softmax(out)

def eval_model(model, tagged_sentences, word_to_ix, tag_to_ix):
    model.eval()
    err = 0
    total = 0
    # Faire fonction d'évaluation
    """nbsent = len(tagged_sentences)
    for sentence in tagged_sentences:
        x, y = make_tensors_sentence(sentence, word_to_ix, tag_to_ix)
        x = ag.Variable(x)
        out = model(x)
        _, out = torch.max(out, 1)
        for i in range(out.data.size()[0]):
            err += 1 if out.data[i] != y[i] else 0 
            total += 1
    print("Test on %s sentence, (err / total) : %s / %s" % (nbsent, err, total))"""