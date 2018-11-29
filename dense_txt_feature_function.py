from numpy import *
import scipy.spatial.distance as dist #距离scipy公式
from fuzzywuzzy import fuzz
import jieba
#from Utils import tokenize_cn
from fuzzywuzzy import process
# The form of an input will be sentence/vector/embeddings

# Chinese tokenize
def tokenize_cn(text):
    return jieba.lcut(text)

# txtFeat1 : Length distance
def func_txtFeat1(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Absolute difference and ratio of two length
    """
    absvalue = abs(len(s1)-len(s2))
    ratio = len(s1)/len(s2)
    return absvalue,ratio

# txtFeat2 : Character distance
def func_txtFeat2(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Absolute difference and proportion of two sentence's characters
    """
    absvalue = abs(len(set(s1.replace(' ','')))-len(set(s2.replace(' ',''))))
    proportion = len(set(s1.replace(' ','')))/len(set(s2.replace(' ','')))
    return absvalue,proportion

# txtFeat3 : Word distance
def func_txtFeat3(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Absolute difference and proportion of two sentence's word
    """
    if lang=='eng' or lang=='id':
        absvalue = abs(len(s1.split(' '))-len(s2.split(' ')))
        ratio = len(s1.split(' '))/len(s2.split(' '))
        return absvalue, ratio
    elif lang=='cn':
        absvalue = abs(len(tokenize_cn(s1))-len(tokenize_cn(s2))) # tokenize_cn is a chinese tokenizer
        ratio = len(tokenize_cn(s1))/len(tokenize_cn(s2))
        return absvalue, ratio

# txtFeat4 : Word distance #1
def func_txtFeat4(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Jaccard distance
    """
    list1 =[]
    list2 =[]

    if lang=='eng' or lang=='id':
        list1 = s1.split()
        list2 = s2.split()
    elif lang=='cn':
        list1 = tokenize_cn(s1)
        list2 = tokenize_cn(s2)

    wordlist = list(set(list1).union(set(list2)))
    onehot1 = [0] * len(wordlist)
    onehot2 = [0] * len(wordlist)
    for item in list1:
        for j in range(len(wordlist)):
            if item == wordlist[j]:
                onehot1[j] += 1

    for item in list2:
        for j in range(len(wordlist)):
            if item == wordlist[j]:
                onehot2[j] += 1
    matV = mat([onehot1, onehot2])
    jaccard = dist.pdist(matV, 'jaccard')
    return jaccard[0]

# txtFeat5 : Word distance #2
def func_txtFeat5(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Length of the union
    """
    if lang =='eng' or lang=='id':
        return len(set(s1.split()).union(set(s2.split())))
    elif lang=='cn':
        return len(set(tokenize_cn(s1)).union(set(tokenize_cn(s2))))

# txtFeat6	Word distance #3
def func_txtFeat6(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Length of the intersection
    """
    if lang =='eng' or lang=='id':
        return len(set(s1.split()).intersection(set(s2.split())))
    elif lang=='cn':
        return len(set(tokenize_cn(s1)).intersection(set(tokenize_cn(s2))))

# txtFeat7 : Word distance #4

# txtFeat8 : QRatio
def func_txtFeat8(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: QRatio
    """
    return float(fuzz.QRatio(s1, s2))

# txtFeat9 : WRatio
def func_txtFeat9(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: WRatio
    """
    return float(fuzz.WRatio(s1,s2))

#txtFeat10 : Partial ratio
def func_txtFeat10(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: partial ratio
    """
    return float(fuzz.partial_ratio(s1,s2))

#txtFeat11 : Partial token set ratio
def func_txtFeat11(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Partial token set ratio
    """
    return float(fuzz.partial_token_set_ratio(s1,s2))

#txtFeat12 : Partial token sort ratio
def func_txtFeat12(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: Partial token sort ratio
    """
    return float(fuzz.partial_token_sort_ratio(s1,s2))

#txtFeat13 : Token set ratio
def func_txtFeat13(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: token set ratio
    """
    return float(fuzz.token_set_ratio(s1,s2))

#txtFeat14 : Token sort ratio
def func_txtFeat14(s1,s2,lang='eng'):
    """
    :param s1: sentence1
    :param s2: sentence2
    :param lang: language of the sentences
    :return: token sort ratio
    """
    return float(fuzz.token_sort_ratio(s1,s2))


def make_txtFeat(s1,s2,lang='cn'):
    function_list=  list(func_txtFeat1(s1,s2,lang))+\
                    list(func_txtFeat2(s1,s2,lang))+\
                    list(func_txtFeat3(s1,s2,lang))+\
                    [func_txtFeat4(s1,s2,lang)]+\
                    [func_txtFeat5(s1,s2,lang)]+\
                    [func_txtFeat6(s1,s2,lang)]+\
                    [func_txtFeat8(s1,s2,lang)]+\
                    [func_txtFeat9(s1,s2,lang)]+\
                    [func_txtFeat10(s1,s2,lang)]+\
                    [func_txtFeat11(s1,s2,lang)]+\
                    [func_txtFeat12(s1,s2,lang)]+\
                    [func_txtFeat13(s1,s2,lang)]+\
                    [func_txtFeat14(s1,s2,lang)]
    return function_list

print (make_txtFeat('今天天气很好','今天的天气不错啊'))

