from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile
import numpy as np
from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.
  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.
  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """Preapre all necessary files that are required for the training.
    Args:
      data_dir: the folder to store the processed file
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.

  to_vocab_path = os.path.join(data_dir, "vocab.to")
  from_vocab_path =  os.path.join(data_dir,"vocab.from")
  create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size, tokenizer)
  create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = os.path.join(data_dir,"train.tgt.ids")
  from_train_ids_path = os.path.join(data_dir,"train.src.ids")
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)

  # Create token ids for the development data.
  to_dev_ids_path = os.path.join(data_dir,"dev.tgt.ids")
  from_dev_ids_path = os.path.join(data_dir,"dev.src.ids")
  data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)


def prepare_test_data(data_dir, from_test_path, from_vocab_path, tokenizer=None):
  
  from_test_ids_path = os.path.join(data_dir,"test.src.ids")
  data_to_token_ids(from_test_path, from_test_ids_path, from_vocab_path, tokenizer)

  return from_test_ids_path

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(data_set); data_set contains a list of
      (source, target) pairs read from the provided data files;
      source and target are lists of token-ids.
  """  
  data_set = []
  max_len = 0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        assert len(source_ids) == len(target_ids)
        max_len = len(source_ids) if len(source_ids) > max_len else max_len
        data_set.append([source_ids, target_ids])
        source, target = source_file.readline(), target_file.readline()
  return data_set, max_len  


def read_test_data(source_path):
    """Read data from source and target files.
    """
    data_set = []
    max_len = 0
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source:
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            max_len = len(source_ids) if len(source_ids) > max_len else max_len
            data_set.append(source_ids)
            source = source_file.readline()
    return data_set, max_len


def get_vocab_info(cache_dir):
    from_vocab_path = os.path.join(cache_dir, "vocab.from")
    to_vocab_path = os.path.join(cache_dir, "vocab.to")
    from_vocab_size = get_real_vocab_size(from_vocab_path)
    to_vocab_size = get_real_vocab_size(to_vocab_path)
    return from_vocab_path,to_vocab_path,from_vocab_size,to_vocab_size


def get_real_vocab_size(vocab_path):
    n = 0
    f = open(vocab_path)
    for line in f:
        n+=1
    f.close()
    return n


def ids_to_tokens(lines, to_vocab_path, dest_path):
    d = []
    f = open(to_vocab_path)
    for line in f:
        line = line.strip()
        d.append(line)
    f.close()

    f = open(dest_path,'w')
    for line in lines:
        sent = " ".join([d[x] for x in line[:-1]])
        f.write(sent+"\n")
    f.close()
    
    
def read_word_vec(word_vec_path, cache_dir):
    from_vocab_path = os.path.join(cache_dir, "vocab.from")
    vocab_dict = {}
    with open(from_vocab_path, 'r') as fin:
      for word_id, word in enumerate(fin.readlines()):
        vocab_dict[word] = word_id
    np.random.seed(10)
    vocab_size = len(vocab_dict)
    with open(word_vec_path, 'r') as fin :
        [num, dim] = [int(ele) for ele in fin.readline().strip().split()]
        words = []
        vectors = []
        for line in fin.readlines():
            tmp = line.strip().split()
            words.append(tmp[0])
            vector = [float(ele) for ele in tmp[1:dim+1]]
            vectors.append(vector)

    vectors = np.array(vectors)
    var = np.var(vectors, axis=0)
    word_vectors = np.empty([vocab_size, dim])
    for i in range(dim):
        high = np.sqrt(3) * var[i]
        low = -1 * high
        word_vectors[:,i] = np.random.uniform(low, high, size=vocab_size)
    word_vectors[0,:] = np.zeros(dim)
    
    word_position = [vocab_dict.get(word, -1) for word in words]
    pre_trained_word_num = sum([ele != -1 for ele in word_position])
    for i, pos in enumerate(word_position) :
        if pos == -1 :
            continue
        word_vectors[pos] = vectors[i]
    return word_vectors, pre_trained_word_num
