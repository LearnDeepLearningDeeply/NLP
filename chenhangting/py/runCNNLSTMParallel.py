from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib import learn
import logging
import data_utils
from cnnLstmParallelModel import ParallelModel

import data_iterator
from data_iterator import DataIterator
from tensorflow.python.client import timeline

from summary import ModelSummary, variable_summaries

import argparse

############################
######## MARK:FLAGS ########
############################
parser = argparse.ArgumentParser(description="Sequence to Sequence Using Tensorflow.")
# mode
parser.add_argument("--mode", type=str, default="TRAIN", help="TRAIN|DECODE")

# datasets, paths, and preprocessing
parser.add_argument("--model_dir", type=str, default="./model", help="model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
parser.add_argument("--train_path_from", type=str, default="./train.src", help="the absolute path of raw source train file.")
parser.add_argument("--dev_path_from", type=str, default="./dev.src", help="the absolute path of raw source dev file.")
parser.add_argument("--test_path_from", type=str, default="./test.src", help="the absolute path of raw source test file.")
parser.add_argument("--word_vec_path", type=str, default="./word2vec.txt", help="the absolute path of word vectors file.")

parser.add_argument("--train_path_to", type=str, default="./train.tgt", help="the absolute path of raw target train file.")
parser.add_argument("--dev_path_to", type=str, default="./dev.tgt.tgt", help="the absolute path of raw target dev file.")
parser.add_argument("--test_path_to", type=str, default="./test.tgt", help="the absolute path of raw target test file.")

parser.add_argument("--decode_output", type=str, default="./test.output", help="beam search decode output.")

# tuning hypers
parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate.")
parser.add_argument("--learning_rate_decay_factor", type=float, default=0.83, help="Learning rate decays by this much.")
parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
parser.add_argument("--keep_prob", type=float, default=0.8, help="dropout rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training/evaluation.")
parser.add_argument("--dilation_rate",type=int,default=1,help="the dilatation of dilated convoluational kernel")

parser.add_argument("--from_vocab_size", type=int, default=10000, help="from vocabulary size.")
parser.add_argument("--to_vocab_size", type=int, default=10000, help="to vocabulary size.")

parser.add_argument("--size", type=int, default=128, help="Size of each model layer.")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model.")
parser.add_argument("--n_epoch", type=int, default=50, help="Maximum number of epochs in training.")

parser.add_argument("--L", type=int, default=2000, help="max length")
parser.add_argument("--patience", type=int, default=10, help="exit if the model can't improve for $patence / 2 epochs")

# devices
parser.add_argument("--N", type=str, default="000", help="GPU layer distribution: [input_embedding, lstm, output_embedding]")

# training parameter
parser.add_argument("--withAdagrad", type=bool, default=True, help="withAdagrad.")
parser.add_argument("--fromScratch", type=bool, default=True, help="fromScratch.")
parser.add_argument("--saveCheckpoint", type=bool, default=False, help="save Model at each checkpoint.")
parser.add_argument("--profile", type=bool, default=False, help="False = no profile, True = profile")

# GPU configuration
parser.add_argument("--allow_growth", type=bool, default=False, help="allow growth")

# Summary
parser.add_argument("--with_summary", type=bool, default=False, help="with_summary")
parser.add_argument("--data_cache_dir", type=str, default="data_cache", help="data_cache")

FLAGS = parser.parse_args()

def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)


def mylog_section(section_name):
    mylog("======== {} ========".format(section_name)) 

def mylog_subsection(section_name):
    mylog("-------- {} --------".format(section_name)) 

def mylog_line(section_name, message):
    mylog("[{}] {}".format(section_name, message))


def get_device_address(s):
    add = []
    if s == "":
        for i in xrange(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add

def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)


def log_flags():
    mylog_section("FLAGS")
    for attr, value in sorted(FLAGS.__dict__.items()):
        mylog("{}={}".format(attr.upper(), value))

def create_model(session, run_options, run_metadata, max_len):
    devices = get_device_address(FLAGS.N)
    dtype = tf.float32
    model = ParallelModel(FLAGS.size,
                     FLAGS.real_vocab_size_from,
                     FLAGS.real_vocab_size_to,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     max_len,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad = FLAGS.withAdagrad,
                     dropoutRate = FLAGS.keep_prob,
                     dtype = dtype,
                     devices = devices,
                     run_options = run_options,
                     run_metadata = run_metadata,
                     word_vector = FLAGS.word_vector,
                     dilation_rate=FLAGS.dilation_rate,
                     )

    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)

    if FLAGS.mode == "DECODE":
        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():

    # Read Data
    mylog_section("READ DATA")

    from_train = None
    to_train = None
    from_dev = None
    to_dev = None
    
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_cache_dir,
        FLAGS.train_path_from,
        FLAGS.train_path_to,
        FLAGS.dev_path_from,
        FLAGS.dev_path_to,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)

    train_data, train_max_len = data_utils.read_data(from_train,to_train)
    dev_data, dev_max_len = data_utils.read_data(from_dev,to_dev)
    max_len = max(train_max_len, dev_max_len)
    _,_,real_vocab_size_from,real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    if FLAGS.word_vector:
    	word_embedding, _ = data_utils.read_word_vec(FLAGS.word_vec_path, FLAGS.data_cache_dir)
    else:
    	word_embedding = None

    train_n_tokens = np.sum([len(items[1]) for items in train_data])
    train_total_size = len(train_data)
    dev_total_size = len(dev_data)

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)
    steps_per_checkpoint = int(steps_per_epoch / 2)
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))


    mylog_section("IN TENSORFLOW")
    
    with tf.Graph().as_default():
        tf.set_random_seed(23)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
        config.gpu_options.allow_growth = FLAGS.allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        with tf.Session(config=config) as sess:
            
            # runtime profile
            if FLAGS.profile:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None
    
            mylog_section("MODEL/SUMMARY/WRITER")
    
            mylog("Creating Model.. (this can take a few minutes)")
            model = create_model(sess, run_options, run_metadata, max_len)
    
            if FLAGS.with_summary:
                mylog("Creating ModelSummary")
                modelSummary = ModelSummary()
    
                mylog("Creating tf.summary.FileWriter")
                summaryWriter = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir , "train.summary"), sess.graph)
    
            mylog_section("All Variables")
            show_all_variables() #convenient for checking model variables
    
            # Data Iterators
            mylog_section("Data Iterators")
    
            dite = DataIterator(model, train_data, batch_size)
            
            iteType = 0
            if iteType == 0:
                mylog("Itetype: withRandom")
                ite = dite.next_random()
            elif iteType == 1:
                mylog("Itetype: withSequence")
                ite = dite.next_sequence()
            
            # statistics during training
            step_time, loss = 0.0, 0.0
            correct = 0
            current_step = 0
            previous_losses = []
            high_acc = -1
            high_acc_step = 0
            steps_per_report = 30
            n_targets_report = 0
            report_time = 0
            n_valid_sents = 0
            n_valid_words = 0
            patience = FLAGS.patience
            
            mylog_section("TRAIN")
    
            
            while current_step < total_steps:
                
                # start
                start_time = time.time()
                
                # data and train
                source_inputs, source_lengths, target_outputs = ite.next()

                L, unary_scores, transition_matrix = model.step(sess, source_inputs, target_outputs, source_lengths, word_embedding)
                
                # loss and time
                step_time += (time.time() - start_time) / steps_per_checkpoint
    
                _, correct_labels = CRF_viterbi_decode(unary_scores, transition_matrix, source_lengths, target_outputs)
                correct += correct_labels
                loss += L
                
                current_step += 1
                n_valid_sents += np.sum(np.sign(source_lengths))
                n_valid_words += np.sum(source_lengths)
    
                # for report
                report_time += (time.time() - start_time)
                n_targets_report += np.sum(source_lengths)
    
                if current_step % steps_per_report == 0:
                    sect_name = "STEP {}".format(current_step)
                    msg = "StepTime: {:.2f} sec Speed: {:.2f} targets/s Total_targets: {}".format(report_time/steps_per_report, n_targets_report*1.0 / report_time, train_n_tokens)
                    mylog_line(sect_name,msg)
    
                    report_time = 0
                    n_targets_report = 0
                    
    
                    # Create the Timeline object, and write it to a json
                    if FLAGS.profile:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('timeline.json', 'w') as f:
                            f.write(ctf)
                        exit()

                
                if current_step % steps_per_checkpoint == 0:
    
                    i_checkpoint = int(current_step / steps_per_checkpoint)
                    
                    # train_acc
                    loss = loss / n_valid_words
                    train_acc = correct * 1.0 / n_valid_words
                    learning_rate = model.learning_rate.eval()
                                    
                    # dev_acc
                    dev_loss, dev_acc = evaluate(sess, model, dev_data, word_embedding)
    
                    # report
                    sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                    msg = "Learning_rate: {:.4f} Dev_acc: {:.4f} Train_acc: {:.4f}".format(learning_rate, dev_acc, train_acc)
                    mylog_line(sect_name, msg)
    
                    if FLAGS.with_summary:
                        # save summary
                        _summaries = modelSummary.step_record(sess, train_acc, dev_acc)
                        for _summary in _summaries:
                            summaryWriter.add_summary(_summary, i_checkpoint)
                    
                    # save model per checkpoint
                    if False: # FLAGS.saveCheckpoint
                        checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                        s = time.time()
                        model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph = False)
                        msg = "Model saved using {:.2f} sec at {}".format(time.time()-s, checkpoint_path)
                        mylog_line(sect_name, msg)
                        
                    # save best model
                    if dev_acc > high_acc:
                        patience = FLAGS.patience
                        high_acc = dev_acc
                        high_acc_step = current_step
                        checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                        s = time.time()
                        model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph = False)
                        msg = "Model saved using {:.2f} sec at {}".format(time.time()-s, checkpoint_path)
                        mylog_line(sect_name, msg)
                    
                    else:
                        patience -= 1
    
                    if patience <= 0:
                        mylog("Training finished. Running out of patience.")
                        break
                    
    
                    # Save checkpoint and zero timer and loss.
                    step_time, loss, n_valid_sents, n_valid_words, correct = 0.0, 0.0, 0, 0, 0
                

def evaluate(sess, model, data_set, word_embedding):
    # Run evals on development set and print their perplexity/loss or accuracy.
    dropoutRateRaw = FLAGS.keep_prob

    start_id = 0
    loss = 0.0
    n_steps = 0
    n_valids = 0
    n_correct = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, batch_size)
    ite = dite.next_sequence()

    for sources, lengths, outputs in ite:
        L, unary_scores, transition_matrix= model.step(sess, sources, outputs, lengths, word_embedding, forward_only = True)
        _, correct_labels = CRF_viterbi_decode(unary_scores, transition_matrix, lengths, outputs)
        n_correct += correct_labels
        loss += L
        n_steps += 1
        n_valids += np.sum(lengths)

    loss = loss/(n_valids)
    acc = n_correct * 1.0 /(n_valids)


    return loss, acc


def CRF_viterbi_decode(unary_scores, transition_matrix, lengths, target_outputs=None):
    correct_labels = 0
    viterbi_seqs = []
    for i, (unary_score, length) in enumerate(zip(unary_scores, lengths)):
        unary_score = unary_score[:length]
        if length:
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(unary_score, transition_matrix)
        else:
            viterbi_seq = []
        viterbi_seqs.append(viterbi_seq)
        if target_outputs is not None:
            target = target_outputs[i][:length]
            correct_labels += np.sum(np.equal(viterbi_seq, target))
    return viterbi_seqs, correct_labels


def decode(ans=False):
    mylog_section("READ DATA")

    from_vocab_path, to_vocab_path, real_vocab_size_from, real_vocab_size_to = data_utils.get_vocab_info(FLAGS.data_cache_dir)
    FLAGS.real_vocab_size_from = real_vocab_size_from
    FLAGS.real_vocab_size_to = real_vocab_size_to

    from_test = data_utils.prepare_test_data(FLAGS.data_cache_dir, FLAGS.test_path_from, from_vocab_path)
    if ans:
        to_test = os.path.join(FLAGS.data_cache_dir, "test.tgt.ids")
        data_utils.data_to_token_ids(FLAGS.test_path_to, to_test, to_vocab_path)
        test_data, test_max_len = data_utils.read_data(from_test, to_test)
    else:
        test_data, test_max_len = data_utils.read_test_data(from_test)

    test_total_size = len(test_data)

    # reports
    mylog_section("REPORT")
    mylog("from_vocab_size: {}".format(FLAGS.from_vocab_size))
    mylog("to_vocab_size: {}".format(FLAGS.to_vocab_size))
    mylog("DECODE:")
    mylog("total: {}".format(test_total_size))
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    mylog_section("IN TENSORFLOW")
    with tf.Session(config=config) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata, test_max_len)
                
        mylog_section("All Variables")
        show_all_variables()
 
        #sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size

        mylog_section("Data Iterators")
        dite = DataIterator(model, test_data, batch_size)
        ite = dite.next_sequence(test = (not ans))

        i_sent = 0

        mylog_section("DECODING")
        results = []
        sources = []
        targets = []

        n_valids = 0
        n_correct = 0

        if ans:
            n_valids = 0
            n_correct = 0
            for inputs, lengths, outputs in ite:
    
                mylog("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
                i_sent += 1
    
                L, unary_scores,transition_matrix = model.step(sess, inputs, outputs, lengths, forward_only = True)
                viterbi_seqs, correct_labels = CRF_viterbi_decode(unary_scores, transition_matrix, lengths, outputs)
                results += viterbi_seqs
                n_correct += correct_labels
                n_valids += np.sum(lengths)

                mylog("Correct Labels / Total Labels: {} / {}".format(correct_labels, np.sum(lengths)))
                n_valids += np.sum(lengths)
                n_correct += correct_labels
    
                sources += [inputs[i][:length] for i, length in enumerate(lengths)]
                targets += [outputs[i][:length] for i, length in enumerate(lengths)]

        else:
            outputs = np.zeros([batch_size, test_max_len], dtype=int)
            for inputs, lengths in ite:
                # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
                # positions: [4]
                mylog("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
                i_sent += 1
                   
                L, unary_scores,transition_matrix = model.step(sess, inputs, outputs, lengths, forward_only = True)
                viterbi_seqs, _ = CRF_viterbi_decode(unary_scores, transition_matrix, lengths)
                results += viterbi_seqs
                mylog("LOSS: {}".format(L))
    
                sources += [inputs[i][:length] for i, length in enumerate(lengths)]
    
            # do the following convert:
            # inputs: [[pad_id],[1],[2],[pad_id],[pad_id],[pad_id]]
            # positions:[2]
        data_utils.ids_to_tokens(results, to_vocab_path, FLAGS.decode_output)
        data_utils.ids_to_tokens(sources, from_vocab_path, FLAGS.decode_output+'.src')
        if ans:
            data_utils.ids_to_tokens(targets, to_vocab_path, FLAGS.decode_output+'.tgt')
            msg = "Test_acc: {:.4f}".format(n_correct * 1.0 /(n_valids))
            mylog(msg)
        mylog("Decoding finished.")


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def parsing_flags():
    # saved_model

    FLAGS.data_cache_dir = os.path.join(FLAGS.model_dir, "data_cache")
    FLAGS.saved_model_dir = os.path.join(FLAGS.model_dir, "saved_model")
    FLAGS.summary_dir = FLAGS.saved_model_dir
    FLAGS.word_vector = True if os.path.exists(FLAGS.word_vec_path) else False

    mkdir(FLAGS.model_dir)
    mkdir(FLAGS.data_cache_dir)
    mkdir(FLAGS.saved_model_dir)
    mkdir(FLAGS.summary_dir)

    # for logs
    log_path = os.path.join(FLAGS.model_dir,"log.{}.txt".format(FLAGS.mode))
    filemode = 'w' if FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode = filemode)
    
    log_flags()

def main(_):
    
    parsing_flags()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.N)
    
    if FLAGS.mode == "TRAIN":
        train()

    
    if FLAGS.mode == 'DECODE':        
        FLAGS.batch_size = 1
        FLAGS.word_vector = False
        answer = True if os.path.exists(FLAGS.test_path_to) else False
        decode(answer)
    
    logging.shutdown()
    
if __name__ == "__main__":
    tf.app.run()
