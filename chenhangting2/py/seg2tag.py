import random
import re
import sys

def raw2segList(sent):
	words = sent.split()
	return [re.sub(r'/[a-zA-Z]+', '', word) for word in words]

def word2tag(word): #gbk
	assert len(word) % 2 == 0 and len(word) >= 2
	if len(word) == 2:
		return ['S']
	else:
		return ['B'] + ['M'] * int((len(word) - 4) / 2) + ['E']

def raw2tag(sent):
	src = []
	tgt = []
	segList = raw2segList(sent)
	for word in segList:
		if len(word) % 2 != 0:
			print(word);print(len(word))
		assert len(word) % 2 == 0
		if len(word) == 0:
			continue
		src += [word[i:i+2] for i in range(0,len(word),2)]
		tgt += word2tag(word)

	return src, tgt

def loadCorpus(filename):
	with open(filename, 'r',encoding='gbk') as fin:
		corpus = []
		line = fin.readline().strip()
		while line:
			line = re.sub(r'\d{8}-\d{2}-\d{3}-\d{3}/m\s+', '', line)
			line = re.sub(r'\[|\][a-zA-Z]+\s', '', line)
			line = re.sub(r'\xa1\xa3/w', '\xa1\xa3/w\n', line) # period
			line = re.sub(r'\xa3\xa1/w', '\xa3\xa1/w\n', line) # exclamation mark
			line = re.sub(r'\xa3\xbf/w', '\xa3\xbf/w\n', line) # question mark
			line = re.sub(r'\xa3\xbb/w', '\xa3\xbb/w\n', line) # semicolon
			line = re.sub(r'\xa3\xac/w', '\xa3\xac/w\n', line) # comma
			line = re.sub(r'\xa1\xa2/w', '\xa1\xa2/w\n', line) # slight-pause mark
			line = re.sub(r'\xa3\xba/w', '\xa3\xba/w\n', line) # colon
			line = re.sub(r'\xa3\xa8/w', '\xa3\xa8/w\n', line) # left parenthesis
			line = re.sub(r'\xa3\xa9/w', '\xa3\xa9/w\n', line) # righ parwnthesis
			sents = line.split('\n')
			corpus += sents
			line = fin.readline()
	return corpus

def processCorpus(corpus):
	srcs = []
	tgts = []
	for sent in corpus:
		src, tgt = raw2tag(sent)
		if len(src) == 0:
			continue
		srcs.append(src)
		tgts.append(tgt)
	return srcs, tgts

def splitData(src, tgt, train_size = 0.6, valid_size = 0.2):
	assert len(src) == len(tgt)
	data_size = len(src)
	random.seed(data_size)
	rand_index = list(range(data_size))
	random.shuffle(rand_index)
	valid_start_index = int(data_size * train_size)
	test_start_index = int(data_size * (train_size + valid_size))
	train_src = []
	train_tgt = []
	valid_src = []
	valid_tgt = []
	test_src = []
	test_tgt = []
	for i in range(valid_start_index):
		train_src.append(src[rand_index[i]])
		train_tgt.append(tgt[rand_index[i]])
	for i in range(valid_start_index, test_start_index):
		valid_src.append(src[rand_index[i]])
		valid_tgt.append(tgt[rand_index[i]])
	for i in range(test_start_index, data_size):
		test_src.append(src[rand_index[i]])
		test_tgt.append(tgt[rand_index[i]])
	return train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt

def writeData(data, filename):
	with open(filename, 'w',encoding='gbk') as fout, open('../data/longSent.txt', 'a+',encoding='gbk') as flong, open('../data/shortSent.txt', 'a+',encoding='gbk') as fst:
		for line in data:
			if len(line) <= 1:
				fst.write(' '.join(line) + '\n')
			else:
				if len(line) <= 100:
					fout.write(' '.join(line) + '\n')
				else:
					flong.write(' '.join(line) + '\n')

def buildVocab(data):
	vocab = set()
	for sent in data:
		for word in sent:
			vocab.add(word)
	return vocab

if __name__ == '__main__':
	filename = sys.argv[1]
	corpus = loadCorpus(filename)
	srcs, tgts = processCorpus(corpus)
	train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt = splitData(srcs, tgts)
	writeData(train_src, '../data/train.src')
	writeData(train_tgt, '../data/train.tgt')
	writeData(valid_src, '../data/valid.src')
	writeData(valid_tgt, '../data/valid.tgt')
	writeData(test_src, '../data/test.src')
	writeData(test_tgt, '../data/test.tgt')
	vocab = buildVocab(srcs)
	print('vocabulary size is %d'%len(vocab))

