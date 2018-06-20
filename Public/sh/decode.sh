PY=../py/run.py
MODEL_DIR=../model/model_segHan
TRAIN_PATH_FROM=../data/train.src
DEV_PATH_FROM=../data/valid.src
TEST_PATH_FROM=../data/test.src
TRAIN_PATH_TO=../data/train.tgt
DEV_PATH_TO=../data/valid.tgt
TEST_PATH_TO=../data/test.tgt
DECODE_OUTPUT=../data/test.output
WORD_VEC=../data/WordVectors.txt

python $PY --mode DECODE --model_dir $MODEL_DIR \
	--test_path_from $TEST_PATH_FROM --test_path_to $TEST_PATH_TO \
    --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
	--N 123 --decode_output $DECODE_OUTPUT
	