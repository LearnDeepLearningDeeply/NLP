import sys
B = 'B'
E = 'E'
M = 'M'
S = 'S'
TAG = [B, E, M, S]

def tag2seg(sent, tags):
	assert len(sent) == len(tags)
	seg = ""
	for word, tag in zip(sent, tags):
		if tag not in TAG:
			tag == S
		if tag == E or tag == S:
			word += '  '
		seg += word
	return seg.strip()


def t2s(f1, f2, f3):
	with open(f1, 'r',encoding='gbk') as fin1, open(f2, 'r',encoding='gbk') as fin2, open(f3, 'w',encoding='gbk') as fin3:
		sent, tags = fin1.readlines(), fin2.readlines()
		for s, t in zip(sent, tags):
			s = s.strip().split()
			t = t.strip().split()
			seg = tag2seg(s, t)
			fin3.write(seg+'\n')

if __name__ == '__main__':
	assert len(sys.argv) == 4
	f1, f2, f3 = sys.argv[1:]
	t2s(f1, f2, f3)
