import re
import random
import sys
random.seed(0)

alpha_full=r'[ｑｗｅｒｔｙｕｉｏｐａｓｄｆｇｈｊｋｌｚｘｃｖｂｎｍＱＷＥＲＴＹＵＩＯＰＡＳＤＦＧＨＪＫＬＺＸＣＶＢＮＭ]+'
alpha = r'[ｑｗｅｒｔｙｕｉｏｐａｓｄｆｇｈｊｋｌｚｘｃｖｂｎｍＱＷＥＲＴＹＵＩＯＰＡＳＤＦＧＨＪＫＬＺＸＣＶＢＮＭa-zA-z]+'
pattern2pattern=[
(alpha_full, '_ENG'),
(r'[a-zA-z]+', '_ENG'),
(r'[0-9]{8}-[0-9]{2}-[0-9]{3}-[0-9]{3}', '_ID'),
(r'0','０'),
(r'1','１'),
(r'2','２'),
(r'3','３'),
(r'4','４'),
(r'5','５'),
(r'6','６'),
(r'7','７'),
(r'8','８'),
(r'9','９'),
(r'。','。\n'),
(r'！','！\n'),
(r'？','？\n'),
(r'：','：\n'),
(r'，','，\n'),
(r'；','；\n'),
(r'、','、\n'),
]


def writeData(data, filename):
	with open(filename, 'w') as fout:
		for line in data:
			fout.write(' '.join(line) + '\n')

def extract_ENG(string, _id):
	lst = [str(_id+1),]
	pattern = re.compile(alpha)
	lst += pattern.findall(string)
	return lst

def extract_DAT(string, _id):
	lst = [str(_id+1),]
	pattern = re.compile(r'[0-9]{8}-[0-9]{2}-[0-9]{3}-[0-9]{3}')
	lst += pattern.findall(string)
	return lst	

def readTrain(filename):
	with open(filename ,'r') as ftrain:
		srcs, tgts, ids, eng, dat = [], [], [], [], []
		for _id, item in enumerate(ftrain.readlines()):
			eng.append(extract_ENG(item, _id))
			dat.append(extract_DAT(item, _id))
			for p in pattern2pattern:
				item = re.sub(p[0],p[1],item)
			lines = [i for i in item.split('\n') if i.strip() != ""]
			for _id_, line in enumerate(lines):
				words = line.strip().split()
				src, tgt = [], []
				for word in words:
					chars = ' '.join(word)
					chars = re.sub(r'_ I D','_ID',chars)
					chars = re.sub(r'_ E N G','_ENG',chars)
					chars = chars.split()
					src += chars
					assert len(chars) >= 1
					if len(chars) == 1:
						tags = ['S']
					else:
						tags = ['B'] + ['M'] * (len(chars) - 2) + ['E']
					tgt += tags
				assert len(src) == len(tgt)
				srcs.append(src)
				tgts.append(tgt)
				ids.append([str(_id+1), str(_id_ + 1)])
		data_size = len(srcs)
		rand_index = list(range(data_size))
		random.shuffle(rand_index)
		valid_start_index = int(data_size * 0.9)
		train_src, train_tgt, train_ids, = [], [], []
		valid_src, valid_tgt, valid_ids, = [], [], []
		for i in range(valid_start_index):
			train_src.append(srcs[rand_index[i]])
			train_tgt.append(tgts[rand_index[i]])
			train_ids.append(ids[rand_index[i]])
		for i in range(valid_start_index, data_size):
			valid_src.append(srcs[rand_index[i]])
			valid_tgt.append(tgts[rand_index[i]])
			valid_ids.append(ids[rand_index[i]])
		writeData(train_src, 'train.src')
		writeData(train_tgt, 'train.tgt')
		writeData(train_ids, 'train.loc')
		writeData(valid_src, 'valid.src')
		writeData(valid_tgt, 'valid.tgt')
		writeData(valid_ids, 'valid.loc')
		writeData(eng, 'train.eng')
		writeData(dat, 'train.dat')


def readTest(filename):
	with open(filename, 'r') as ftest:
		srcs, tgts, ids, eng, dat = [], [], [], [], []
		for _id, item in enumerate(ftest.readlines()):
			eng.append(extract_ENG(item, _id))
			dat.append(extract_DAT(item, _id))
			for p in pattern2pattern:
				item = re.sub(p[0],p[1],item)
			lines = [i for i in item.split('\n') if i.strip() != ""]
			for _id_, line in enumerate(lines):
				words = line.strip().split()
				src, tgt = [], []
				for word in words:
					chars = ' '.join(word)
					chars = re.sub(r'_ I D','_ID',chars)
					chars = re.sub(r'_ E N G','_ENG',chars)
					chars = chars.split()
					src += chars
					assert len(chars) >= 1
					if len(chars) == 1:
						tags = ['S']
					else:
						tags = ['B'] + ['M'] * (len(chars) - 2) + ['E']
					tgt += tags
				assert len(src) == len(tgt)
				srcs.append(src)
				tgts.append(tgt)
				ids.append([str(_id+1), str(_id_ + 1)])
		writeData(srcs, 'test.src')
		writeData(tgts, 'test.tgt')
		writeData(ids, 'test.loc')
		writeData(eng, 'test.eng')
		writeData(dat, 'test.dat')

def _readTest(filename, outfile):
	with open(filename, 'r') as ftest:
		srcs, ids = [], []
		eng, dat = [], []
		for _id, item in enumerate(ftest.readlines()):
			eng.append(extract_ENG(item, _id))
			dat.append(extract_DAT(item, _id))
			for p in pattern2pattern:
				item = re.sub(p[0],p[1],item)
			lines = [i for i in item.split('\n') if i.strip() != ""]
			for _id_, line in enumerate(lines):
				words = line.strip().split()
				src = []
				for word in words:
					chars = ' '.join(word)
					chars = re.sub(r'_ I D','_ID',chars)
					chars = re.sub(r'_ E N G','_ENG',chars)
					chars = chars.split()
					src += chars
				srcs.append(src)
				ids.append([str(_id+1), str(_id_ + 1)])
		writeData(srcs, outfile+'.src')
		writeData(ids, outfile+'.loc')
		writeData(eng, outfile+'.eng')
		writeData(dat, outfile+'.dat')	

if __name__ == '__main__':
	assert len(sys.argv) == 3
	testFile, outFile = sys.argv[1:]
	_readTest(testFile, outFile)
	
	#_readTest("demo.txt", "demo")
