PYTHONPATH=python
PY=$1
MODEL_DIR=$2
DEV_PATH_FROM=../data/valid.src
DEV_PATH_TO=../data/valid.tgt
DECODE_OUTPUT=$2/valid.output
SEG=../py/t2s.py
SEG_PATH=$2/valid.seg
SCORE=../data/score
SCORE_OPT="../data/pku_training_words.txt ../data/valid_gold.seg"
SCORE_TXT=$2/score_valid.txt

set -e

if [ $# -ne 4 ];then
    echo "$0 python_script model_dir device_id log"
    exit
fi


rm -f $2/data_cache/test.src.ids
rm -f $2/data_cache/test.tgt.ids
rm -f $2/data_cache/dev.src.ids
rm -f $2/data_cache/dev.tgt.ids
    
echo "======== Only decoding validatation set ========" >> $4

$PYTHONPATH -u $PY --mode DECODE --model_dir $MODEL_DIR \
	--test_path_from $DEV_PATH_FROM --test_path_to $DEV_PATH_TO \
    --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
	--N $3 --decode_output $DECODE_OUTPUT >> $4 2>&1
	
$PYTHONPATH $SEG $DEV_PATH_FROM $DECODE_OUTPUT $SEG_PATH >> $4 2>&1

$SCORE $SCORE_OPT $SEG_PATH > $SCORE_TXT 2>&1

echo "======== Validatation metrics ========" > $4

tail $SCORE_TXT >> $4

