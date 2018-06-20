编译
g++ -o word2vec word2vec_mod.c -lpthread
运行示范
./word2vec -train ../data/train.src -output ../data/WordVectors.txt -save-vocab data/train.vocab -size 128 -epoch 5 -threads 20