import os
import sys
import re
pattern2pattern=[
(r'０','0'),
(r'１','1'),
(r'２','2'),
(r'３','3'),
(r'４','4'),
(r'５','5'),
(r'６','6'),
(r'７','7'),
(r'８','8'),
(r'９','9'),
(r'．','.')
]

B = 'B'
E = 'E'
M = 'M'
S = 'S'
TAG = [B, E, M, S]

def tag2seg(sent, tags):
	assert len(sent) == len(tags),"{}\n{}\n".format(sent,tags)
	seg = ""
	for word, tag in zip(sent, tags):
		if tag not in TAG:
			tag == S
		if tag == E or tag == S:
			word += '  '
		seg += word
	return seg.strip()

def read_data(sent_file_path, tag_file_path, loc_file_path, eng_file_path, dat_file_path, seg_file_path):
	with open(sent_file_path, 'r',encoding='gbk') as fin1, open(tag_file_path, 'r',encoding='gbk') as fin2, open(loc_file_path, 'r',encoding='gbk') as fin3, open(eng_file_path, 'r',encoding='gbk') as fin4, open(dat_file_path, 'r',encoding='gbk') as fin5, open(seg_file_path, 'w',encoding='gbk') as fout:
		sent, tags, loc = fin1.readline(), fin2.readline(), fin3.readline()
		engs, dats = fin4.readlines(), fin5.readlines()
		segs = ""
		while sent and tags and loc:
			sent = sent.strip().split()
			tags = tags.strip().split()
			loc = int(loc.strip().split()[-1])
			seg = tag2seg(sent, tags)
			if loc == 1:
				segs += '\n' + seg
			else:
				segs += '  ' + seg
			sent, tags, loc = fin1.readline(), fin2.readline(), fin3.readline()
		segs = segs.strip().split('\n')
		assert len(segs) == len(engs)
		assert len(segs) == len(dats)
		for seg, eng, dat in zip(segs, engs, dats):
			eng = eng.strip().split()
			dat = dat.strip().split()
			if len(eng) > 1:
				seg = seg.split('_ENG')
				assert len(seg) == len(eng),"{}\n{}\n".format(seg,eng)
				s = seg[0]
				for i in range(1, len(seg)):
					s += eng[i] + seg[i]
				seg = s
			if len(dat) > 1:
				seg = seg.split('_DAT')
				assert len(seg) == len(dat)
				s = seg[0]
				for i in range(1, len(seg)):
					s += dat[i] + seg[i]
				seg = s
			for p in pattern2pattern:
				seg = re.sub(p[0],p[1],seg)
			fout.write(seg+'\n')				


if __name__ == '__main__':
	assert len(sys.argv) == 7
	sent_file_path, tag_file_path, loc_file_path, eng_file_path, dat_file_path, seg_file_path = sys.argv[1:]
	assert os.path.exists(sent_file_path) and os.path.exists(tag_file_path) and os.path.exists(loc_file_path) and os.path.exists(eng_file_path) and os.path.exists(dat_file_path)
	read_data(sent_file_path, tag_file_path, loc_file_path, eng_file_path, dat_file_path, seg_file_path)
