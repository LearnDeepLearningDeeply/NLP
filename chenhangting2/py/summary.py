import tensorflow as tf



class ModelSummary:
    def __init__(self):
        with tf.name_scope("ModelSummary"):
            with tf.device("/cpu:0"):
                self.train_acc = tf.placeholder(tf.float32, shape = (), name = "train_acc")
                self.dev_acc = tf.placeholder(tf.float32, shape = (), name = "dev_acc")
                self.summary_train_acc = tf.summary.scalar("train_acc", self.train_acc)
                self.summary_dev_acc = tf.summary.scalar("dev_acc", self.dev_acc)
            
    def step_record(self, sess, train_acc, dev_acc):
        input_feed = {}
        input_feed[self.train_acc.name] = train_acc
        input_feed[self.dev_acc.name] = dev_acc
        
        output_feed = [self.summary_train_acc, self.summary_dev_acc ]
        
        outputs = sess.run(output_feed, input_feed)
        return outputs


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
