DECODE_OUTPUT=../data/test.output
SEG=../py/tag2seg.py
SEG_PATH=../data/test.seg

python $SEG $DECODE_OUTPUT.src $DECODE_OUTPUT $SEG_PATH
echo "Done!"