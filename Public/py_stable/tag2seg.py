import os
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
			word += ' '
		seg += word
	return seg.strip()

def read_data(sent_file_path, tag_file_path, seg_file_path):
	with open(sent_file_path, 'r',encoding='gbk') as fin1, open(tag_file_path, 'r',encoding='gbk') as fin2, open(seg_file_path, 'w',encoding='gbk') as fout:
		sent, tags = fin1.readline(), fin2.readline()
		while sent and tags:
			sent = sent.strip().split()
			tags = tags.strip().split()
			seg = tag2seg(sent, tags)
			fout.write(seg+'\n')
			sent, tags = fin1.readline(), fin2.readline()

if __name__ == '__main__':
	assert len(sys.argv) == 4
	sent_file_path, tag_file_path, seg_file_path = sys.argv[1:]
	assert os.path.exists(sent_file_path) and os.path.exists(tag_file_path)
	read_data(sent_file_path, tag_file_path, seg_file_path)

