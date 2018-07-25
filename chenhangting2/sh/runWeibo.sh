class=mix
PYTHONPATH=python
PY=$1
MODEL_DIR=$2
MODEL_NAME=$5
DEV_PATH_FROM=../weibo/processed2/$class.src
DEV_PATH_TO=../weibo/wrong.path
DECODE_OUTPUT=../weibo/result/$5/$class.output
SEG=../data/tag2seg.py
SEG_PATH=../weibo/result/$5/$class.seg
opt="../weibo/processed2/$class.loc ../weibo/processed2/$class.eng ../weibo/processed2/$class.dat"

set -e

if [ $# -ne 5 ];then
    echo "$0 python_script model_dir device_id log model_name"
    exit
fi

rm -f ../weibo/result/$5/*
rm -f $2/data_cache/test.src.ids
rm -f $2/data_cache/test.tgt.ids
rm -f $2/data_cache/dev.src.ids
rm -f $2/data_cache/dev.tgt.ids


$PYTHONPATH -u $PY --mode DECODE --model_dir $MODEL_DIR \
	--test_path_from $DEV_PATH_FROM --test_path_to $DEV_PATH_TO \
    --from_vocab_size 5000 --to_vocab_size 10 --size 128 \
	--N $3 --decode_output $DECODE_OUTPUT > $4 2>&1 
	
$PYTHONPATH $SEG $DEV_PATH_FROM $DECODE_OUTPUT $opt $SEG_PATH >> $4 2>&1

#cp $SEG_PATH ${SEG_PATH}.copy
#sed -i "s/  _PAD//g" ${SEG_PATH}.copy
#sed -i "s/_PAD//g" ${SEG_PATH}.copy
