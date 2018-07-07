PYTHONPATH=~/pythonlhr
PY=$1
MODEL_DIR=$2
TRAIN_PATH_FROM=../data/train.src
DEV_PATH_FROM=../data/valid.src
TEST_PATH_FROM=../data/test.src
TRAIN_PATH_TO=../data/train.tgt
DEV_PATH_TO=../data/valid.tgt
TEST_PATH_TO=../data/test.tgt
DECODE_OUTPUT=$2/test.output
WORD_VEC=$2/WordVectors.txt
SEG=../py/tag2seg.py
SEG_PATH=$2/test.seg

set -e

if [ $# -ne 4 ];then
    echo "$0 python_script model_dir device_id log"
    exit
fi

echo "======== Training Phase ========" > $4
$PYTHONPATH -u $PY --mode TRAIN --model_dir $MODEL_DIR \
    --train_path_from $TRAIN_PATH_FROM --dev_path_from $DEV_PATH_FROM \
    --train_path_to $TRAIN_PATH_TO --dev_path_to $DEV_PATH_TO \
    --batch_size 128 --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
    --n_epoch 50 --saveCheckpoint True --learning_rate 0.5 --keep_prob 0.5 \
    --N $3 --word_vec_path $WORD_VEC >> $4 2>&1

echo "======== Test Phase ========" >> $4
$PYTHONPATH -u $PY --mode DECODE --model_dir $MODEL_DIR \
	--test_path_from $TEST_PATH_FROM --test_path_to $TEST_PATH_TO \
    --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
	--N $3 --decode_output $DECODE_OUTPUT >> $4 2>&1
	
echo "======== Reorganize Test Corpus ========" >> $4
$PYTHONPATH $SEG $DECODE_OUTPUT.src $DECODE_OUTPUT $SEG_PATH >> $4 2>&1
echo "======== Finished ========" >> $4
