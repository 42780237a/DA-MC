import tensorflow as tf
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.ops.lookup_ops import IdTableWithHashBuckets


class BaseModel(object):
    def __init__(self, hparam, export=False, with_char=False):
        self.hparam = hparam

        self.Embedding = None
        self.embed_matrix = None
        self.init_embedding = None

        if export:
            table = HashTable(TextFileIdTableInitializer(filename='assets/vocab',
                                                         key_column_index=0,
                                                         value_column_index=1, vocab_size=None, delimiter='\t',
                                                         name='table'), default_value=0)
            self.q1_string = tf.placeholder(tf.string, [None, self.hparam.seq_length], name='string_input1')
            self.q2_string = tf.placeholder(tf.string, [None, self.hparam.seq_length], name='string_input2')
            self.premise = table.lookup(self.q1_string)
            self.hypothesis = table.lookup(self.q2_string)
            
            if with_char:
                char_table = HashTable(TextFileIdTableInitializer(filename='assets/char_vocab',
                                                                  key_column_index=0,
                                                                  value_column_index=1,
                                                                  vocab_size=None,
                                                                  delimiter='\t',name='char_table'),default_value=0)
                self.ch1_string = tf.placeholder(tf.string, [None, self.hparam.char_seq_length], name='char_string_input1')
                self.ch2_string = tf.placeholder(tf.string, [None, self.hparam.char_seq_length], name='char_string_input1')
                self.ch1 = char_table.lookup(self.ch1_string)
                self.ch2 = char_table.lookup(self.ch2_string)
            else:
                self.ch1,self.ch2 = None,None
        else:
            self.premise = tf.placeholder(tf.int32, [None, self.hparam.seq_length], 'premise')
            self.hypothesis = tf.placeholder(tf.int32, [None, self.hparam.seq_length], 'hypothesis')            
            self.ch1, self.ch2 = None, None

        self.premise_mask, self.hypothesis_mask = None, None
        self.y = None
        self.pred = None
        self.logits = None
        self.dropout_keep_prob = None

        self.loss = None
        self.train_op = None
        self.is_training = None

        self.output_w = None
        self.output_b = None

        self.lsf_q1, self.lsf_q2 = None, None

        self.random_size = None
