import torch as th

def make_tensors_data(data, batch_size):
	tmp = data.tolist()
	l = len(tmp)
	res = []
	while l != 0:
		bs = batch_size if l >= batch_size else l % batch_size
		t = []
		for _ in range(bs):
			img = tmp.pop()
			t.append(img)
		t = th.FloatTensor(t)
		if th.cuda.is_available():
			t = t.cuda()
		res.append(t)
		l = len(tmp)
	return res

def make_tensors_label(labels_list, batch_size):
	tmp = labels_list
	l = len(tmp)
	res = []
	while l != 0:
		bs = batch_size if l >= batch_size else l % batch_size
		t = []
		for _ in range(bs):
			img = tmp.pop()
			t.append(img)
		t = th.LongTensor(t)
		if th.cuda.is_available():
			t = t.cuda()
		res.append(t)
		l = len(tmp)
	return res