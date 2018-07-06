PYTHONPATH=~/pythonlhr
PY=../py/run.py
MODEL_DIR=../model/model_segHan
TEST_PATH_FROM=../weibo/processed/pos6000.txt
TEST_PATH_TO=../weibo/wrong.path
DECODE_OUTPUT=../weibo/decode/pos6000.txt
WORD_VEC=../data/WordVectors.txt

$PYTHONPATH $PY --mode DECODE --model_dir $MODEL_DIR \
	--test_path_from $TEST_PATH_FROM --test_path_to $TEST_PATH_TO \
    --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
	--N 123 --decode_output $DECODE_OUTPUT
	
