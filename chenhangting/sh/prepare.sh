PY=../py/seg2tag.py
CORPUS=../data/ChineseCorpus199801.txt
echo "Generating train/valid/test (sentence, tag) pairs"
python $PY $CORPUS
echo "Training word vectors using train set"
../word2vec/word2vec -train ../data/train.src -output ../data/WordVectors.txt -save-vocab ../data/train.vocab -size 128 -epoch 5 -threads 20