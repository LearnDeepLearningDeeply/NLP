# -*- coding: utf-8 -*-
"""
@date: Created on 2018/7/5

@notes: This script is to process raw weibo Corpus to *.scr, which can be accepted by run.py.
    usage : python raw_weibo_corpus processed_weibo_corpus
"""

from __future__ import print_function
import re
import sys
import time

pattern2enter=[
    r'//',
    # '@霍思燕 '
    r'@[^ :：。\.]+ ',r'@[^ :：。\.]+:',r'@[^ :：。\.]+：',r'@[^ :：。\.]+。',r'@[^ :：。\.]+\.',r'@[^ :：。\.]+'
]
pattern2blank=[
    # '[开心]'
    r'\[[^\]]+\]',r'【[^】]+】',
    # '#Sweet Morning#'
    r'#.*#',
    r'http://[a-zA-Z0-9/\.]+'
    # '《名流巨星》'
    # r'\xa1b6.*\xa1b7',
]
pattern2pattern=[
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
    (r',','，'),
    (r'!','！'),
    (r'\.','。'),
    (r':','：'),
    # 瞎几把判断
    # (r'\'','“'),
    (r'\(','（'),
    (r'\)','）'),
    (r'~','～'),
    (r'/','／'),
    (r'\?','？'),
]
# pattern2space=[]

def loadCorpus(line,showFlag=False):
    line=line.strip()
    print(line)
    # 替换数字中的小数点为全角
    line=re.sub(r'(?P<d1>\d+)\.(?P<d2>\d+)','\g<d1>．\g<d2>',line)
    for p in pattern2blank:
        line=re.sub(p,'',line)
    for p in pattern2enter:
        line=re.sub(p,'\n',line)
    for p in pattern2pattern:
        line=re.sub(p[0],p[1],line)
    line=re.sub(r'([a-zA-Z][a-zA-Z]+[a-zA-Z，！。：（）～／？])?','',line)
    # line=re.sub(r'[a-zA-Z]','',line)
    line=re.sub(' ','',line)

    line=line.split('\n')
    line=[l for l in line if len(l)>0]

    if(showFlag):time.sleep(3)
    print(line)
    print('\n')
    return line

def writeCorpus(lineList,f):
    for line in lineList:
        if(len(line)<=2):return
        else:f.write(' '.join(line)+'\n')

if __name__ == '__main__':
    filefrom = sys.argv[1]
    fileto=sys.argv[2]
    showFlag=False
    with open(filefrom,'r') as ffrom, open(fileto,'w') as fto:
        for line in ffrom:
            lineList=loadCorpus(line,showFlag=showFlag)
            writeCorpus(lineList,fto)
