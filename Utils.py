'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import numpy as np
import os
from datetime import timedelta
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import pickle
import time
import json
import pandas as pd
from dense_txt_feature_function import make_txtFeat
from dense_vect_feature_function import make_vecFeat_v2
import jieba

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
#CATEGORIE_ID = {'entailment' : 0, 'neutral' : 1, 'contradiction' : 2}
CATEGORIE_ID = {'entailment' : 1, 'contradiction' : 0}  # for sentence2Index()
REVERSED_CATEGORIE_ID = {1:'entailment' , 0:'contradiction' }   # for convert_form()


# Chinese tokenize 
def tokenize_cn(text):
    return jieba.lcut(text)


# check data form whether like :s1<tab>s2<tab>label
def check_data_form(file_path, write_file=False):
    '''
    param: file_path
    return: number of bad line
    '''
    with open(file_path,encoding='utf8') as f:
        i = 0
        bad=0
        new_list= []
        for line in f:
            text_list= line.strip().split('\t')
            if len(text_list)!=3:
                print ('length error in line %s : %s' % (str(i),line))
                bad+=1
            elif text_list[0]=='' or text_list[1]=='' :
                print ('Empty sentence in line %s : %s' % (str(i),line))
                bad+=1
            elif (str(text_list[2].strip()) not in set(['0','1'])):
                print ('label value error in line %s : %s' % (str(i),line))
                bad+=1
            else :
                new_list.append(line)

            i+=1
    if write_file:
        file_out =open(file_path,mode='w')
        for l in new_list:
            file_out.write(l)
        file_out.close()
    return bad

def recall_K(y_pred_prob, y_true, index_list,K):

    '''
    :param y_pred_prob y_pred_prob=[[0.04,0.96],[0.08,0.92], ... ,[0.99, 0.01]]
    :param y_true: list of the true value in (0,1) :[0,1, ... ,0]
    :param index list of each line : [[0,10],...,[100,115]]
    :param K : threshold
    :return : the recall@K score
    '''
    count=0
    df=pd.DataFrame({'y_pred_prob_1':[v[1] for v in y_pred_prob],'y_true':y_true})
    denominator = len(index_list)
    for i in index_list:
        df_block=df.loc[i[0]:i[1]-1].reset_index(drop=True)

        if df_block.loc[df_block.loc[:,'y_true']==1].shape[0]==0:
            denominator -= 1
        elif df_block.loc[df_block.loc[:,'y_true']==0].shape[0]==0:
            count+=1
        else:

            max_label_1 = (max(df_block.loc[df_block.loc[:,'y_true']==1]['y_pred_prob_1']))
            prob_0 = df_block.loc[df_block.loc[:,'y_true']==0]['y_pred_prob_1']
            #print (sum(prob_0>=max_label_1))
            print ("num prob0>max_pro1 :%d" % sum(prob_0>=max_label_1))
            if sum(prob_0>=max_label_1)<K:
                count+=1

    return count/denominator


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)




def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))

# init embeddings randomly
def init_embeddings(vocab, embedding_dims):
    """
    :param vocab: word nums of the vocabulary
    :param embedding_dims: dimension of embedding vector
    :return: randomly init embeddings with shape (vocab, embedding_dims)
    """
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(scale=0.01, size = (len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)

# load pre-trained embeddings
def load_embeddings(path, vocab):
    """
    :param path: path of the pre-trained embeddings file
    :param vocab: word nums of the vocabulary
    :return: pre-trained embeddings with shape (vocab, embedding_dims)
    """
    with open(path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab, embedding_dims)
    for word, id in vocab.items():
        #print(word, id)
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)

# normalize the word embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis = 1).reshape((-1, 1))
    return embeddings / norms

# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams

# time cost
def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds = int(round(diff)))

# build vocabulary according the training data
def build_vocab(dataPath, vocabPath, threshold = 0, lowercase = True):
    """
    :param dataPath: path of training data file
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :param lowercase: boolean, lower words or not
    """
    cnt = Counter()
    with open(dataPath, mode='r', encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('||')

                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                #print (l1, l2)
                words1 = tokenize_cn(l1)
                for word in list(words1):
                    cnt[word] += 1
                words2 = tokenize_cn(l2)
                #print (words1, words2)
                for word in list(words2):
                    cnt[word] += 1
            except:
                pass
    cntDict = [item for item in cnt.items() if item[1] >= threshold]
    cntDict = sorted(cntDict, key=lambda d: d[1], reverse=True)
    wordFreq = ['||'.join([word, str(freq)]) for word, freq in cntDict]
    with open(vocabPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(wordFreq) + '\n')
    print('Vacabulary is stored in : {}'.format(vocabPath))

# load vocabulary
def load_vocab(vocabPath, threshold = 0):
    """
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :return: vocab: vocabulary dict {word : index}
    """
    vocab = {}
    index = 2
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    with open(vocabPath, encoding='utf-8') as f:
        for line in f:
            items = [v.strip() for v in line.split('||')]
            if len(items) != 2:
                print('Wrong format: ', line)
                continue
            word, freq = items[0], int(items[1])
            if freq >= threshold:
                vocab[word] = index
                index += 1
    return vocab

# data preproceing, convert words into indexes according the vocabulary
def sentence2Index(dataPath, vocabDict, embeddings, maxLen = 100, lowercase = True, write_features=False):
    """
    :param dataPath: path of data file
    :param vocabDict: vocabulary dict {word : index}
    :param maxLen: max length of sentence, if a sentence longer than maxLen, cut off it
    :param lowercase: boolean, lower words or not
    :return: s1Pad: padded sentence1
             s2Pad: padded sentence2
             s1Mask: actual length of sentence1
             s2Mask: actual length of sentence2
             features: dense features
    """
    s1List, s2List, labelList = [], [], []
    s1Mask, s2Mask = [], []
    s1Embed, s2Embed = [], []
    txt_features = []
    vec_features = []
    featuresPath = "./RWS/clean data/features.txt"
    with open(dataPath, mode='r', encoding='utf-8') as f:
        for line in f:
            try:
                if write_features:
                    featuresFile=open(featuresPath, "a+")
                l, s1, s2 = [v.strip() for v in line.strip().split('||')]
                #print (l,s1,s2)
                #if lowercase:
                    #s1, s2 = s1.lower(), s2.lower()
                s1 = [v.strip() for v in tokenize_cn(s1)]
                s2 = [v.strip() for v in tokenize_cn(s2)]
                if len(s1) > maxLen:
                    s1 = s1[:maxLen]
                if len(s2) > maxLen:
                    s2 = s2[:maxLen]
                txtFeat=make_txtFeat(' '.join(s1), ' '.join(s2))
                txt_features.append(txtFeat)

                if l in CATEGORIE_ID:
                    #labelList.append([CATEGORIE_ID[l]])
                    labelList.append(CATEGORIE_ID[l])
                    s1_vocab_index=[vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1]
                    s2_vocab_index=[vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2]
                    s1List.append(s1_vocab_index)
                    s2List.append(s2_vocab_index)
                    s1Mask.append(len(s1))
                    s2Mask.append(len(s2))

                    s1Embed=embeddings[s1_vocab_index]
                    s2Embed=embeddings[s2_vocab_index]
                    vecFeat=make_vecFeat_v2(s1Embed, s2Embed)
                    #print (vecFeat)
                    vec_features.append(vecFeat)
                if write_features:
                    content_to_write= '||'.join([line.strip(),','.join(map(str,txtFeat)),','.join(map(str,vecFeat))])+'\n'
                    featuresFile.write(content_to_write)
                    featuresFile.close()
            except:
                ValueError('Input Data Value Error!')

    s1Pad, s2Pad = pad_sequences(s1List, maxLen, padding='post', value=0), pad_sequences(s2List, maxLen, padding='post', value=0)
    s1Mask = np.asarray(s1Mask, np.int32) # len of premise
    s2Mask = np.asarray(s2Mask, np.int32) # len of hypothesis
    labelList = np.asarray(labelList, np.int32)
    #print (labelList)
    #features=[txt_features[i]+vec_features[i] for i in range(len(features))]
    features = vec_features
    features = np.asarray(features, np.float32)
    print (features)
    print ("Length of features: %s"% len(features) )

    return s1Pad, s1Mask, s2Pad, s2Mask, labelList, features

def sentence2Index_v2(dataPath, vocabDict, maxLen = 100, lowercase = True):
    """
    :param dataPath: path of data+feature file
    :param vocabDict: vocabulary dict {word : index}
    :param maxLen: max length of sentence, if a sentence longer than maxLen, cut off it
    :param lowercase: boolean, lower words or not
    :return: s1Pad: padded sentence1
             s2Pad: padded sentence2
             s1Mask: actual length of sentence1
             s2Mask: actual length of sentence2
             features: dense features
    """
    s1List, s2List, labelList = [], [], []
    s1Mask, s2Mask = [], []
    txt_features = []
    vec_features = []

    with open(dataPath, mode='r', encoding='utf-8') as f:
        for line in f:
            try:

                l, s1, s2, txtFeat, vecFeat= [v.strip() for v in line.strip().split('||')]
                txtFeat = [float(i) for i in txtFeat.split(',')]
                vecFeat = [float(i) for i in vecFeat.split(',')]
                #if lowercase:
                #    s1, s2 = s1.lower(), s2.lower()
                s1 = [v.strip() for v in tokenize_cn(s1)]
                s2 = [v.strip() for v in tokenize_cn(s2)]
                if len(s1) > maxLen:
                    s1 = s1[:maxLen]
                if len(s2) > maxLen:
                    s2 = s2[:maxLen]

                txt_features.append(txtFeat)
                vec_features.append(vecFeat)

                if l in CATEGORIE_ID:
                    #labelList.append([CATEGORIE_ID[l]])
                    labelList.append(CATEGORIE_ID[l])
                    s1_vocab_index=[vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1]
                    s2_vocab_index=[vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2]
                    s1List.append(s1_vocab_index)
                    s2List.append(s2_vocab_index)
                    s1Mask.append(len(s1))
                    s2Mask.append(len(s2))

                #content_to_write= '||'.join([line.strip(),','.join(map(str,txtFeat)),','.join(map(str,vecFeat))])+'\n'

            except:
                ValueError('Input Data Value Error!')

    s1Pad, s2Pad = pad_sequences(s1List, maxLen, padding='post', value=0), pad_sequences(s2List, maxLen, padding='post', value=0)
    s1Mask = np.asarray(s1Mask, np.int32) # len of premise
    s2Mask = np.asarray(s2Mask, np.int32) # len of hypothesis
    labelList = np.asarray(labelList, np.int32)
    #features=[txt_features[i]+vec_features[i] for i in range(len(features))]
    features = vec_features
    features = np.asarray(features, np.float32)
    print ("Length of features: %s"% len(features) )

    return s1Pad, s1Mask, s2Pad, s2Mask, labelList, features


# generator : generate a batch of data
def next_batch(premise, premise_mask, hypothesis, hypothesis_mask, y, features, batchSize = 64, shuffle = True):
    """
    :param premise_mask: actual length of premise
    :param hypothesis_mask: actual length of hypothesis
    :param shuffle: boolean, shuffle dataset or not
    :param features : dense features
    :return: generate a batch of data (premise, premise_mask, hypothesis, hypothesis_mask, label)
    """
    sampleNums = len(premise)
    batchNums = int((sampleNums - 1) / batchSize) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(sampleNums))
        premise = premise[indices]
        premise_mask = premise_mask[indices]
        hypothesis = hypothesis[indices]
        hypothesis_mask = hypothesis_mask[indices]
        y = y[indices]
        features = features[indices]

    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, sampleNums)
        #print ("Num of 1: %s ,Num of 0: %s, Total num: %s" %(sum([int(i[1]) for i in y[startIndex : endIndex]]), sum([int(i[0]) for i in y[startIndex : endIndex]]), (y[startIndex : endIndex].shape[0])) )
        yield (premise[startIndex : endIndex], premise_mask[startIndex : endIndex],
               hypothesis[startIndex : endIndex], hypothesis_mask[startIndex : endIndex],
               y[startIndex : endIndex], features[startIndex : endIndex])

# convert RWS dataset from format : (sentence1 \t sentence2 \t int_label) to format : (str_label || sentence1 || sentence2)
def convert_form(inputPath, outputPath):
    #inputPath='./testset1_v1_3_new.txt'
    #outputPath='./testset1_v1_3_converted.txt'
    """
    :param inputPath: path of RWS dataset file
    :param outputPath: path of output
    """
    fout = open(outputPath, 'w',encoding='utf8')
    with open(inputPath,encoding='utf8') as fin:
        i = 0
        for line in fin:
            text_list= line.strip().split('\t')
            if len(text_list)!=3:
                print ('length error in line %s : %s' % (str(i),line))
            elif (str(text_list[2].strip()) not in set(['0','1'])):
                print ('label value error in line %s : %s' % (str(i),line))
            else:
                print('||'.join([REVERSED_CATEGORIE_ID[int(text_list[2].strip())], text_list[0].strip(), text_list[1].strip()]), file = fout)

            i += 1
            if i % 10000 == 0:
                print(i)
    print('Source data has been converted from "{0}" to "{1}".'.format(inputPath, outputPath))

#Convert RWS's test data from "q1*q2<tab>q3*q4<tab>q5" to "q1<tab>q2<tab>1"
def convert_form_v2(inputPath,outputPath): #inpuPath="./testset2_v1_3.txt"
    '''
    :param inputPath: path of input file
    :param outputPath: path of output file
    :return index list of each line : [[0,10],...,[100,115]]
    '''
    with open(inputPath, mode='r', encoding='utf-8') as iF:
        new_file=[]
        index_list=[]
        begin=0
        end=0
        for line in iF:
            count=0
            line_list=line.strip().split('*')
            #print (df_test2.loc[i,0],df_test2.loc[i,1],df_test2.loc[i,2],df_test2.loc[i,3])
            for q2 in  line_list[1].strip().split('\t'):
                if q2!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q2,'1']))
            for q3 in  line_list[2].strip().split('\t'):
                if q3!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q3,'0']))
            for q4 in  line_list[3].strip().split('\t'):
                if q4!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q4,'0']))

            end+=count
            index_list.append([begin,end])
            begin=end
    with open(outputPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(new_file))
    return index_list


# convert SNLI dataset from json to txt (format : gold_label || sentence1 || sentence2)
def convert_data(jsonPath, txtPath):
    """
    :param jsonPath: path of SNLI dataset file
    :param txtPath: path of output
    """
    fout = open(txtPath, 'w')
    with open(jsonPath) as fin:
        i = 0
        cnt = {key : 0 for key in CATEGORIE_ID.keys()}
        cnt['-'] = 0
        for line in fin:
            text = json.loads(line)
            cnt[text['gold_label']] += 1
            print('||'.join([text['gold_label'], text['sentence1'], text['sentence2']]), file = fout)

            i += 1
            if i % 10000 == 0:
                print(i)

    for key, value in cnt.items():
        print('#{0} : {1}'.format(key, value))
    print('Source data has been converted from "{0}" to "{1}".'.format(jsonPath, txtPath))

# convert embeddings from txt to format : (embeddings, vocab_dict)
def convert_embeddings(srcPath, dstPath):
    """
    :param srcPath: path of source embeddings
    :param dstPath: path of output
    """
    vocab = {}
    id = 0
    wrongCnt = 0
    with open(srcPath, 'r', encoding = 'utf-8') as fin:
        lines = fin.readlines()
        wordNums = len(lines)
        line = lines[0].strip().split()
        vectorDims = len(line) - 1
        embeddings = np.zeros((wordNums, vectorDims), dtype = np.float32)
        for line in lines:
            items = line.strip().split()
            if len(items) != vectorDims + 1:
                wrongCnt += 1
                continue
            if items[0] in vocab:
                wrongCnt += 1
                continue
            vocab[items[0]] = id
            embeddings[id] = [float(v) for v in items[1:]]
            id += 1

        embeddings = embeddings[0 : id, ]
        with open(dstPath, 'wb') as fout:
            pickle.dump([embeddings, vocab], fout)

        print('valid embedding nums : {0}, embeddings shape : {1},'
              ' wrong format embedding nums : {2}, total embedding nums : {3}'.format(len(vocab),
                                                                                      embeddings.shape,
                                                                                      wrongCnt,
                                                                                      wordNums))
        print('Original embeddings has been converted from {0} to {1}'.format(srcPath, dstPath))

# print log info on SCREEN and LOG file simultaneously
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None

# print all used hyper-parameters on both SCREEN an LOG file
def print_args(args, log_file):
    """
    :Param args: all used hyper-parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log("------------- HYPER PARAMETERS -------------", file = log_file)
    for a in argsList:
        print_log("%s: %s" % (a[0], str(a[1])), file = log_file)
    print("-----------------------------------------", file = log_file)
    return None

if __name__ == '__main__':
    # dataset preprocessing
    #if not os.path.exists('./SNLI/clean data/'):
    #    os.makedirs('./SNLI/clean data/')

    #convert_data('./SNLI/raw data/snli_1.0_train.jsonl', './SNLI/clean data/train.txt')
    #convert_data('./SNLI/raw data/snli_1.0_dev.jsonl', './SNLI/clean data/dev.txt')
    #convert_data('./SNLI/raw data/snli_1.0_test.jsonl', './SNLI/clean data/test.txt')

    # embedding preprocessing
    # convert_embeddings('./SNLI/raw data/glove.840B.200d.txt', './SNLI/clean data/embeddings.pkl')

    # vocabulary preprocessing
    #build_vocab('./SNLI/clean data/train.txt', './SNLI/clean data/vocab.txt')

    if not os.path.exists('./RWS/clean data/'):
        os.makedirs('./RWS/clean data/')
    #print ("Bad line numbers: %s in trainset " % check_data_form('./RWS/raw data/en_train.txt'))
    #print ("Bad line numbers: %s in devset" % check_data_form('./RWS/raw data/testset1_mix_zh.txt'))
    #convert_form('./RWS/raw data/en_train.txt', './RWS/clean data/en_train.txt')
    #convert_form('./RWS/raw data/testset1_mix_zh.txt', './RWS/clean data/en_test.txt')
    #convert_form('./RWS/raw data/RWS-dev.txt', './RWS/clean data/en_dev.txt')

    # embedding preprocessing
    #convert_embeddings('./RWS/raw data/crawl-100d-2M.vec', './RWS/clean data/embeddings.pkl')

    # vocabulary preprocessing
    build_vocab('./RWS/clean data/en_train.txt', './RWS/clean data/vocab.txt')



