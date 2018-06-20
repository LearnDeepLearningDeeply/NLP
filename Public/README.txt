(1)语料：存放在 data/
ChineseCorpus199801.txt:原始语料文件

(2)脚本：存放在 sh/
prepare.sh: 生成train/valid/test，并从train set中训练word2vec

train.sh: 只训练模型，关键参数包括
--n_epoch：迭代次数
--size 128：隐含层大小
--num_layers：隐含层大小(目前只支持一层BLSTM,多层需要修改代码)
--keep_prob：dropout
--N：指定使用GPU
其余参数不建议随意修改

decode.sh: 只对测试集解码，加载model/model_segHan/saved_model中最后一次保存的模型(最佳)
保证命令行中的参数和训练模型的隐含层大小、隐含层大小这些参数一致
如果不提供正确答案，最终在data/segHan中生成两个文件，test.output和test.output.src
test.output和test.output.src是一一对应的(标签序列，字序列)，test.output和test.src并不一一对应(不是按照原始数据存放顺序解码的)
如果提供正确答案，最终在data/segHan中生成三个文件，test.output，test.output.src和test.output.tgt
多出来的test.output.tgt是正确答案，顺序和test.output，test.output.src一致，以供参考或计算正确率

seg.sh：从(字序列，标签序列)中生成分词结果，存放在data/中

run.sh：训练模型后解码，结合了train.sh，decode.sh和seg.sh中的命令

(3)分词源代码：存放在 py/
seg2tag.py: 
运行命令示例 python seg2tag.py ChineseCorpus199801.txt
从原始语料中生成所有的(字序列，标签序列)，并按照6:2:2比例划分train/valid/test
例如：(‘他 责 令 各 部 门 加 快 全 面 建 设 新 都 的 步 伐 。’，‘S B E S B E B E B E B E B E S B E S’)
字序列分别存放在train.src，valid.src，test.src(视为源序列，source)
标签序列分别存放在train.tgt，valid.tgt，test.tgt(视为目标序列，target)
以上六个文件是模型的输入文件，已经存放在指定路径中，即：data/
提取过程中，序列长度小于2，大于100的被剔除，剔除的样本存放在shortSent.txt/longSent.txt

tag2seg.py:
运行命令示例参见 sh/seg.sh
从(字序列，标签序列)中生成分词结果，存放在data/中

data.utils.py: 数据预处理模块，功能包括
生成源语言和目标语言的字典，分别存放在model/model_segHan/data_cache/中的vocab.from和vocab.to，第1行的字符在模型中被编码为0，第2行的字符在模型中被编码为1，依此类推
生成train.src，valid.src，test.src，train.tgt，valid.tgt，test.tgt对应的编码文件，后缀为.id，同样存放在model/model_segHan/data_cache/

data_iterator.py：从整个数据中生成批数据提供给模型(通过yield函数实现)

seqModel.py：定义网络结构(1-layer BLSTM + CRF loss)

run.py：定义读取数据、训练模型、解码过程

其余py文件暂时没有用

(4)word2vec源代码：存放在 word2vec/

(5)remarks.txt：记录了编程过程中一些注意点