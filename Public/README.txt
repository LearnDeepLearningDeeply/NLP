(1)���ϣ������ data/
ChineseCorpus199801.txt:ԭʼ�����ļ�

(2)�ű�������� sh/
prepare.sh: ����train/valid/test������train set��ѵ��word2vec

train.sh: ֻѵ��ģ�ͣ��ؼ���������
--n_epoch����������
--size 128���������С
--num_layers���������С(Ŀǰֻ֧��һ��BLSTM,�����Ҫ�޸Ĵ���)
--keep_prob��dropout
--N��ָ��ʹ��GPU
������������������޸�

decode.sh: ֻ�Բ��Լ����룬����model/model_segHan/saved_model�����һ�α����ģ��(���)
��֤�������еĲ�����ѵ��ģ�͵��������С���������С��Щ����һ��
������ṩ��ȷ�𰸣�������data/segHan�����������ļ���test.output��test.output.src
test.output��test.output.src��һһ��Ӧ��(��ǩ���У�������)��test.output��test.src����һһ��Ӧ(���ǰ���ԭʼ���ݴ��˳������)
����ṩ��ȷ�𰸣�������data/segHan�����������ļ���test.output��test.output.src��test.output.tgt
�������test.output.tgt����ȷ�𰸣�˳���test.output��test.output.srcһ�£��Թ��ο��������ȷ��

seg.sh����(�����У���ǩ����)�����ɷִʽ���������data/��

run.sh��ѵ��ģ�ͺ���룬�����train.sh��decode.sh��seg.sh�е�����

(3)�ִ�Դ���룺����� py/
seg2tag.py: 
��������ʾ�� python seg2tag.py ChineseCorpus199801.txt
��ԭʼ�������������е�(�����У���ǩ����)��������6:2:2��������train/valid/test
���磺(���� �� �� �� �� �� �� �� ȫ �� �� �� �� �� �� �� �� ��������S B E S B E B E B E B E B E S B E S��)
�����зֱ�����train.src��valid.src��test.src(��ΪԴ���У�source)
��ǩ���зֱ�����train.tgt��valid.tgt��test.tgt(��ΪĿ�����У�target)
���������ļ���ģ�͵������ļ����Ѿ������ָ��·���У�����data/
��ȡ�����У����г���С��2������100�ı��޳����޳������������shortSent.txt/longSent.txt

tag2seg.py:
��������ʾ���μ� sh/seg.sh
��(�����У���ǩ����)�����ɷִʽ���������data/��

data.utils.py: ����Ԥ����ģ�飬���ܰ���
����Դ���Ժ�Ŀ�����Ե��ֵ䣬�ֱ�����model/model_segHan/data_cache/�е�vocab.from��vocab.to����1�е��ַ���ģ���б�����Ϊ0����2�е��ַ���ģ���б�����Ϊ1����������
����train.src��valid.src��test.src��train.tgt��valid.tgt��test.tgt��Ӧ�ı����ļ�����׺Ϊ.id��ͬ�������model/model_segHan/data_cache/

data_iterator.py�������������������������ṩ��ģ��(ͨ��yield����ʵ��)

seqModel.py����������ṹ(1-layer BLSTM + CRF loss)

run.py�������ȡ���ݡ�ѵ��ģ�͡��������

����py�ļ���ʱû����

(4)word2vecԴ���룺����� word2vec/

(5)remarks.txt����¼�˱�̹�����һЩע���