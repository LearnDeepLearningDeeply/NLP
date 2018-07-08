# Overview
1. 共有三个主要模型，BiLSTM，CNN，CNN-BiLSTM
2. accuary中，train，dev，test分别代表训练、验证、测试的准确率

# BiLSTM
## Structure
1. Input
2. Word Embedding(vocabulary_size * 128)
3. BiLSTM(num_layers * 128units)
4. Output Embedding(256 * tag_size)
5. CRF
6. Tagging

## Params
+ 隐藏层为128个单元固定
+ 隐藏层层数可调
+ 没有dropout和Batch norm

## BiLSTM Results
| No. Layers | Train(%) | Dev(%) | Test(%) |
|:-------|:-------------:|:----------:|:---:|
|   1  | 99.99 | 95.59 | 95.63 |
|   2  | 99.99 | **95.75** | **95.77** |
|   3  | 99.95 | 95.52 | 95.50 |
|   4  | 99.73 | 94.97 | 94.87 |

# CNN
## Structure
1. Input
2. Word Embedding(vocabulary_size * 128)
3. Conv1(kernel_size=3,stride=1,padding='SAME',dilation=1,in_channels=out_channels=128)
4. Conv2(kernel_size=3,stride=1,padding='SAME',dilation=dilation,in_channels=out_channels=128)
5. Conv3(kernel_size=3,stride=1,padding='SAME',dilation=dilation,in_channels=out_channels=128)
6. Conv4(kernel_size=3,stride=1,padding='SAME',dilation=1,in_channels=out_channels=128)
7. Output Embedding(128 * tag_size)
8. CRF
9. Tagging

## Params
+ 卷积核的size，padding，stride均固定
+ 卷积核的dilation rate可调
+ Conv1-4后都接有Batch norm，ReLu
+ Conv2-4前都接有Dropout

## CNN Results
| Dilation rate | Train(%) | Dev(%) | Test(%) |
|:-------|:-------------:|:----------:|:---:|
|   1  | 98.43 | 96.26 | 96.26 |
|   2  | 98.39 | **96.38** | **96.31** |
|   4  | 97.36 | 95.56 | 95.53 |

# CNN-BiLSTM
## Structure
### Parallel
并行经过CNN 和 BiLSTM，最后两者的输出concatenate，再输入至Output Embedding
### Serial
先经过CNN，再经过BiLSTM

## Params
+ CNN和BiLSTM的设置与前面相同
+ BiLSTM的隐层层数设置为2
+ CNN的dilation rate可调

## CNN-BiLSTM Results

| Mixed method | Dilation rate | Train(%) | Dev(%) | Test(%) |
|:-------|:-------------:|:----------:|:---:|:---:|
| Parallel | 1 | 99.48 | 96.49 | 96.37 |
| Parallel | 2 | 99.32 | 96.27 | 96.20 |
| Serial | 1 | 99.18 | **96.82** | **96.78** |
| Serial | 2 | 99.13 | 96.79 | 96.69 |