'''
Created on August 1, 2018
@author : hsiaoyetgun (yqxiao)
Reference : A Decomposable Attention Model for Natural Language Inference (EMNLP 2016)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Utils import print_shape
from BaseModel import BaseModel

class Decomposable(BaseModel):
    def __init__(self, hparam, export=False):
        super(Decomposable, self).__init__(hparam, export)
        self.num_features = 24

        # model init
        self._parameter_init(hparam)
        self._placeholder_init()
        self._add_variables()

        # model operation
        self.logits = self._logits_op()
        self.prob = tf.nn.softmax(self.logits)
        self.loss = self._loss_op(l2_lambda=0.0)
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)
        self._init()

    # init hyper-parameters
    def _parameter_init(self, hparam):
        """
        :param seq_length: max sentence length
        :param n_vocab: word nums in vocabulary
        :param embedding_size: embedding vector dims
        :param hidden_size: hidden dims
        :param attention_size: attention dims
        :param n_classes: nums of output label class
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param optimizer: optimizer of training
        :param l2: l2 regularization constant
        :param clip_value: if gradients value bigger than this value, clip it
        :param num_features: number of dense features
        """
        self.seq_length = hparam.seq_length
        self.n_vocab = hparam.n_vocab
        self.embedding_size = hparam.embedding_size
        self.hidden_size = hparam.hidden_size
        # Note that attention_size is not used in this model
        self.attention_size = hparam.attention_size
        self.n_classes = hparam.n_classes
        self.batch_size = hparam.batch_size
        self.learning_rate = hparam.learning_rate
        self.optimizer = hparam.optimizer
        self.l2 = hparam.l2
        self.clip_value = hparam.clip_value
       

    # placeholder declaration
    def _placeholder_init(self):
        """
        y: actual labels
        premise_mask: actual length of premise sentence
        hypothesis_mask: actual length of hypothesis sentence
        embed_matrix: with shape (n_vocab, embedding_size)
        dropout_keep_prob: dropout keep probability
        is_training: a placeholder to determine training/inference
        :return:
        """
        self.y = tf.placeholder(tf.int32, [None], 'y_true')
        self.features =  tf.placeholder(tf.float32, [None, self.num_features], 'features')
        self.premise_mask = tf.placeholder(tf.int32, [None]) # self.q1_len
        self.hypothesis_mask = tf.placeholder(tf.int32, [None]) # self.q2_len
        self.embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.embedding_size], 'embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name='phase')

    # add variables
    def _add_variables(self):
        """
        Embedding: Variables to hold word embeddings. Untrainable.
        """
        self.Embedding = tf.Variable(tf.truncated_normal([self.n_vocab, self.embedding_size]), dtype=tf.float32, name='Embedding', trainable=False)
        self.init_embedding = self.Embedding.assign(self.embed_matrix)
        self.Embedding = self._projectionBlock(self.Embedding, self.hidden_size, 'Projection')
        print_shape('projected embeddings', self.Embedding)
               
    # load embedding
    def _init(self):
        """
        init_embedding: an assign op to assign pretrained embeddings to the variables
        Embedding: projected Embeddings
        init_model: a global initializer op
        """
        self.init_model = tf.global_variables_initializer()
 
    # build graph
    def _logits_op(self):
        # [batch_size, seq_length, embedding_dim]
        self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
        self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
        print_shape('embeded_left', self.embeded_left)
        print_shape('embeded_right', self.embeded_right)
        
        # [batch_size, seq_length]    
        left_mask = tf.sequence_mask(self.premise_mask, self.seq_length, tf.float32)
        right_mask = tf.sequence_mask(self.hypothesis_mask, self.seq_length, tf.float32)
        print_shape('left_mask', left_mask)
        print_shape('right_mask', right_mask)

        alpha, beta = self._attendBlock('Attend', left_mask, right_mask)
        v_1, v_2 = self._compareBlock(alpha, beta, 'Compare')
        logits = self._aggregateBlock(v_1, v_2, 'Aggregate', left_mask, right_mask, self.features)
        return logits

    # feed forward unit
    def _feedForwardBlock(self, inputs, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor with shape (batch_size, seq_length, embedding_size)
        :param num_units: dimensions of each feed forward layer
        :param scope: scope name
        :return: output: tensor with shape (batch_size, seq_length, num_units)
        """
        with tf.variable_scope(scope, reuse = isReuse):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, num_units, tf.nn.relu, kernel_initializer = initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                results = tf.layers.dense(outputs, num_units, tf.nn.relu, kernel_initializer = initializer)
                return results

    # projection unit
    def _projectionBlock(self, inputs, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor of shape (n_vocab, embedding_size)
        :param num_units: output dim
        :param scope: scope name
        :param isReuse: reuseable or not
        :param initializer: type of initializer
        :return: output: projected tensor (n_vocab, num_units)
        """
        with tf.variable_scope(scope, reuse = isReuse):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            results = tf.layers.dense(inputs, num_units, activation=None, kernel_initializer = initializer)
            return results

    # decomposable attend block ("3.1 Attend" in paper)
    def _attendBlock(self, scope, left_mask, right_mask):
        """
        :param scope: scope name

        embeded_left, embeded_right: tensor with shape (batch_size, seq_length, embedding_size)
        F_a_bar, F_b_bar: output of feed forward layer (F), tensor with shape (batch_size, seq_length, hidden_size)
        attentionSoft_a, attentionSoft_b: using Softmax at two directions, tensor with shape (batch_size, seq_length, seq_length)
        e: attention matrix with mask, tensor with shape (batch_size, seq_length, seq_length)

        :return: alpha: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
                 beta: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        """
        with tf.variable_scope(scope):
            F_a_bar  = self._feedForwardBlock(self.embeded_left, self.hidden_size, 'F')
            F_b_bar = self._feedForwardBlock(self.embeded_right, self.hidden_size, 'F', isReuse = True)
            print_shape('F_a_bar', F_a_bar)
            print_shape('F_b_bar', F_b_bar)

            # e_i,j = F'(a_hat, b_hat) = F(a_hat).T * F(b_hat) (1)
            e_raw = tf.matmul(F_a_bar, tf.transpose(F_b_bar, [0, 2, 1]))

            # mask padding sequence
            #mask = tf.multiply(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))
            #e = tf.multiply(e_raw, mask) + (1.0 - mask)*(-1e9)
            #print_shape('e', e)
            
            right_mask = tf.to_float(tf.expand_dims(right_mask, axis=1))
            e = e_raw + (1.0 - right_mask)*(-1e9)
            beta_attend = tf.nn.softmax(e, dim = -1)
            beta = tf.matmul(beta_attend, self.embeded_right)

            e_raw = tf.transpose(e_raw, [0, 2, 1])
            left_mask = tf.to_float(tf.expand_dims(left_mask, axis=1))
            e = e_raw + (1.0 - left_mask)*(-1e9)
            alpha_attend = tf.nn.softmax(e, dim = -1)
            alpha = tf.matmul(alpha_attend, self.embeded_left)

            # beta = \sum_{j=1}^l_b \frac{\exp(e_{i,j})}{\sum_{k=1}^l_b \exp(e_{i,k})} * b_hat_j
            # alpha = \sum_{i=1}^l_a \frac{\exp(e_{i,j})}{\sum_{k=1}^l_a \exp(e_{k,j})} * a_hat_i (2)
            print_shape('alpha', alpha)
            print_shape('beta', beta)

            return alpha, beta

    # compare block ("3.2 Compare" in paper)
    def _compareBlock(self, alpha, beta, scope):
        """
        :param alpha: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        :param beta: context vectors, tensor with shape (batch_size, seq_length, embedding_size)
        :param scope: scope name

        a_beta, b_alpha: concat of [embeded_premise, beta], [embeded_hypothesis, alpha], tensor with shape (batch_size, seq_length, 2 * embedding_size)

        :return: v_1: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
                 v_2: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        """
        with tf.variable_scope(scope):
            a_beta = tf.concat([self.embeded_left, beta], axis=2)
            b_alpha = tf.concat([self.embeded_right, alpha], axis=2)
            print_shape('a_beta', a_beta)
            print_shape('b_alpha', b_alpha)

            # v_1,i = G([a_bar_i, beta_i])
            # v_2,j = G([b_bar_j, alpha_j]) (3)
            v_1 = self._feedForwardBlock(a_beta, self.hidden_size, 'G')
            v_2 = self._feedForwardBlock(b_alpha, self.hidden_size, 'G', isReuse=True)
            print_shape('v_1', v_1)
            print_shape('v_2', v_2)
            return v_1, v_2

    # composition block ("3.3 Aggregate" in paper)
    def _aggregateBlock(self, v_1, v_2, scope, left_mask, right_mask, dense_features):
        """
        :param v_1: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        :param v_2: compare the aligned phrases, output of feed forward layer (G), tensor with shape (batch_size, seq_length, hidden_size)
        :param scope: scope name

        v1_sum, v2_sum: sum of the compared phrases (axis = seq_length), tensor with shape (batch_size, hidden_size)
        v: concat of v1_sum, v2_sum, tensor with shape (batch_size, 2 * hidden_size)
        ff_outputs: output of feed forward layer (H), tensor with shape (batch_size, hidden_size)

        :return: y_hat: output of a linear layer, tensor with shape (batch_size, n_classes)
        """
        with tf.variable_scope(scope):
            left_mask = tf.to_float(tf.expand_dims(left_mask, axis=2))
            right_mask = tf.to_float(tf.expand_dims(right_mask, axis=2))
            v_1 = v_1 * left_mask
            v_2 = v_2 * right_mask
            
            # v1 = \sum_{i=1}^l_a v_{1,i}
            # v2 = \sum_{j=1}^l_b v_{2,j} (4)
            v1_sum = tf.reduce_sum(v_1, axis=1)
            v2_sum = tf.reduce_sum(v_2, axis=1)
            print_shape('v1_sum', v1_sum)
            print_shape('v2_sum', v2_sum)

            # y_hat = H([v1, v2]) (5)
            v = tf.concat([v1_sum, v2_sum, dense_features], axis=1)
            print_shape('v', v)

            ff_outputs = self._feedForwardBlock(v, self.hidden_size, 'H')
            print_shape('ff_outputs', ff_outputs)

            # compute the logits
            y_hat = tf.layers.dense(ff_outputs, self.n_classes, \
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                                    bias_initializer=tf.contrib.layers.xavier_initializer())
            print_shape('y_hat', y_hat)
            return y_hat

    # calculate classification loss
    def _loss_op(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            #label_true = tf.argmax(self.y, 1, name='label_true')
            label_true = self.y
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # define optimizer
    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op
