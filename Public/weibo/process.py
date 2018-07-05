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
    r'\xa3\xac',r',',
    r'\xa3\xa1',r'!',
    r'…',
    r'[^\d].[^\d]',r'\xa1\xa3',r'[^\d]\xa3\xae[\^d]',
    r'\xa3\xba',r':',
    r'\xa1\xb0',r'\xa1\xb1',r'\'',
    r'\xa3\xa8',r'\xa3\xa9',r'\(',r'\)',
    r'\xa1\xab',r'~',
    r'/',
    r'\xa3\xba',r':',
    r'\?',r'\xa3\xbf',
    r'\xa1\xa2',
]
pattern2blank=[
    # '[开心]'
    r'\[.*\]',r'\xa1\xbe.*\xa1\xbf',
    # '@霍思燕 '
    r'@.* ',r'@.*:',r'@.*\xa3\xba',
    # '#Sweet Morning#'
    r'#.*#',
    # '《名流巨星》'
    r'\xa1b6.*\xa1b7',
]
pattern2pattern=[
    ('0','\xa3\xb0'),
    ('1','\xa3\xb1'),
    ('2','\xa3\xb2'),
    ('3','\xa3\xb3'),
    ('4','\xa3\xb4'),
    ('5','\xa3\xb5'),
    ('6','\xa3\xb6'),
    ('7','\xa3\xb7'),
    ('8','\xa3\xb8'),
    ('9','\xa3\xb9'),
    (r'.','\xa3\xae'),
]
# pattern2space=[]

def loadCorpus(line,showFlag=False):
    if(showFlag):print(line)
    for p in pattern2pattern:
        line=re.sub(p[0],p[1],line)
    for p in pattern2blank:
        line=re.sub(p,'',line)
    for p in pattern2enter:
        line=re.sub(p,'\n',line)

    line=re.sub('[a-zA-Z]','',line)
    line=re.sub(' ','',line)

    line=line.split('\n')
    line=[l for l in line if len(l)!=0]

    if(showFlag):
        time.sleep(5)
        print(line)
    return line

def writeCorpus(lineList,f):
    for line in lineList:
        if(len(line)<=2):return
        else:f.writeline(' '.join(line)+'\n')

if __name__ == '__main__':
    filefrom = sys.argv[1]
    fileto=sys.argv[2]
    showFlag=True
    with open(filefrom,'r') as ffrom, open(fileto,'w') as fto:
        for line in ffrom:
            lineList=loadCorpus(line,showFlag=showFlag)
            writeCorpus(lineList,fto)
