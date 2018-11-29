
import numpy as np
from scipy import spatial
import gensim

emb_path='../data/GoogleNews-vectors-negative300.bin'

class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(emb_path,
                                                                     binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)

# The form of an input will be sentence/vector/embeddings

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

#vecFeat1 : Word mover distance
def func_vecFeat1(s1_token, s2_token, word2vec):
    """
    :param s1_token: sentence1 token
    :param s2_token: sentence2 token
    :return: wmdistance
    """
    # avoid null matrix
    if not len(s1_token) or not len(s2_token):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')
    # wordlist =list(set(s1_token).union(set(s2_token)))

    return word2vec.model.wmdistance(s1_token, s2_token)

#vecFeat2 : Normalized word mover distance

#vecFeat3 : Cosine distance between two vectors
def func_vecFeat3(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2: word_vector2
    :return: Cosine distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return spatial.distance.cosine(vec1_mean,vec2_mean)

#vecFeat4 : Manhattan distance between two vectors
def func_vecFeat4(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector1
    :return: Manhattan distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return np.linalg.norm(vec1_mean - vec2_mean, ord=1)

#vecFeat5 : Jaccard distance between two vectors
def func_vecFeat5(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Jaccard distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return spatial.distance.jaccard(vec1_mean,vec2_mean)

#vecFeat6 : Canberra distance between two vectors
def func_vecFeat6(wordvec1,wordvec2):
    """
    :param wordvec1:wordvector1
    :param wordvec2:wordvector2
    :return:canberra distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return spatial.distance.canberra(vec1_mean,vec2_mean)

#vecFeat7 : Euclidean distance between two vectors
def func_vecFeat7(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Euclidean distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return np.linalg.norm(vec1_mean-vec2_mean)

#vecFeat8 : Minkowski distance between two vectors


#vecFeat9 : Braycurtis distance between two vectors
def func_vecFeat9(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Braycurtis distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1),axis=0)
    vec2_mean = np.mean(np.array(wordvec2),axis=0)

    return spatial.distance.braycurtis(vec1_mean,vec2_mean)

#vecFeat10 : Word mover distance

#vecFeat11 : Normalized word mover distance


#vecFeat12 : Cosine distance between two vectors
def func_vecFeat12(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Cosine distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return spatial.distance.cosine(vec1_l2_norm, vec2_l2_norm)

#vecFeat13 : Manhattan distance between two vectors
def func_vecFeat13(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Manhattan distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return np.linalg.norm(vec1_l2_norm - vec2_l2_norm, ord=1)

#vecFeat14 : Jaccard distance between two vectors
def func_vecFeat14(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Jaccard distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return spatial.distance.jaccard(vec1_l2_norm,vec2_l2_norm)

#vecFeat15 : Canberra distance between two vectors
def func_vecFeat15(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Canberra distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return spatial.distance.canberra(vec1_l2_norm,vec2_l2_norm)

#vecFeat16 : Euclidean distance between two vectors
def func_vecFeat16(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Eucildean distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return np.linalg.norm(vec1_l2_norm-vec2_l2_norm)

#vecFeat17	Minkowski distance between two vectors

#vecFeat18	Braycurtis distance between two vectors
def func_vecFeat18(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Bracurtis distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_mean = np.mean(np.array(wordvec1), axis=0)
    vec2_mean = np.mean(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_mean)
    vec2_l2_norm = normalize(vec2_mean)

    return spatial.distance.braycurtis(vec1_l2_norm,vec2_l2_norm)

#vecFeat19 : Word mover distance


#vecFeat20 : Normalized word mover distance


#vecFeat21 : Cosine distance between two vectors
def func_vecFeat21(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2: word_vector2
    :return: Cosine distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1),axis=0)
    vec2_max = np.max(np.array(wordvec2),axis=0)

    return spatial.distance.cosine(vec1_max,vec2_max)

#vecFeat22 : Manhattan distance between two vectors
def func_vecFeat22(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Manhattan distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    return np.linalg.norm(vec1_max-vec2_max, ord=1)

#vecFeat23 : Jaccard distance between two vectors
def func_vecFeat23(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Jaccard distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    return spatial.distance.jaccard(vec1_max, vec2_max)

#vecFeat24	Canberra distance between two vectors
def func_vecFeat24(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Canberra distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    return spatial.distance.canberra(vec1_max,vec2_max)

#vecFeat25	Euclidean distance between two vectors
def func_vecFeat25(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Euclidean distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    return np.linalg.norm(vec1_max-vec2_max)

#vecFeat26	Minkowski distance between two vectors

#vecFeat27	Braycurtis distance between two vectors
def func_vecFeat27(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Braycurtis distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    return spatial.distance.braycurtis(vec1_max,vec2_max)

#vecFeat28	Word mover distance


#vecFeat29	Normalized word mover distance


#vecFeat30	Cosine distance between two vectors
def func_vecFeat30(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return:Cosine distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return spatial.distance.cosine(vec1_l2_norm, vec2_l2_norm)


#vecFeat31	Manhattan distance between two vectors
def func_vecFeat31(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Manhattan distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return np.linalg.norm(vec1_l2_norm - vec2_l2_norm, ord=1)


#vecFeat32	Jaccard distance between two vectors
def func_vecFeat32(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Jaccard distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return spatial.distance.jaccard(vec1_l2_norm,vec2_l2_norm)

#vecFeat33	Canberra distance between two vectors
def func_vecFeat33(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Canberra distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return spatial.distance.canberra(vec1_l2_norm,vec2_l2_norm)

#vecFeat34	Euclidean distance between two vectors
def func_vecFeat34(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Euclidean
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return  np.linalg.norm(vec1_l2_norm-vec2_l2_norm)


#vecFeat35	Minkowski distance between two vectors

#vecFeat36	Braycurtis distance between two vectors
def func_vecFeat36(wordvec1,wordvec2):
    """
    :param wordvec1:word_vector1
    :param wordvec2:word_vector2
    :return: Braycurtis distance
    """
    if not len(wordvec1) or not len(wordvec2):
        raise Exception('There must have something wrong with your word_vector,perhaps it is null')

    vec1_max = np.max(np.array(wordvec1), axis=0)
    vec2_max = np.max(np.array(wordvec2), axis=0)

    vec1_l2_norm = normalize(vec1_max)
    vec2_l2_norm = normalize(vec2_max)

    return spatial.distance.braycurtis(vec1_l2_norm,vec2_l2_norm)



def make_vecFeat(s1,s2,word2vec,lang='eng'):
    if lang=='eng' or lang=='id':
        s1_token = s1.lower().split()
        s2_token = s2.lower().split()
    elif lang=='cn':
        s1_token = tokenize_cn(s1)
        s2_token = tokenize_cn(s2)

    wordvec1 = []
    for item in s1_token:
        wordvec1.append(word2vec.get(item))
    wordvec2 = []
    for item in s2_token:
        wordvec2.append(word2vec.get(item))

    function_list= [func_vecFeat1(s1_token,s2_token,word2vec),
                    func_vecFeat3(wordvec1,wordvec2),
                    func_vecFeat4(wordvec1,wordvec2),
                    func_vecFeat5(wordvec1,wordvec2),
                    func_vecFeat6(wordvec1,wordvec2),
                    func_vecFeat7(wordvec1,wordvec2),
                    func_vecFeat9(wordvec1,wordvec2),
                    func_vecFeat12(wordvec1,wordvec2),
                    func_vecFeat13(wordvec1,wordvec2),
                    func_vecFeat14(wordvec1,wordvec2),
                    func_vecFeat15(wordvec1,wordvec2),
                    func_vecFeat16(wordvec1,wordvec2),
                    func_vecFeat18(wordvec1,wordvec2),
                    func_vecFeat21(wordvec1,wordvec2),
                    func_vecFeat22(wordvec1,wordvec2),
                    func_vecFeat23(wordvec1,wordvec2),
                    func_vecFeat24(wordvec1,wordvec2),
                    func_vecFeat25(wordvec1,wordvec2),
                    func_vecFeat27(wordvec1,wordvec2),
                    func_vecFeat30(wordvec1,wordvec2),
                    func_vecFeat31(wordvec1,wordvec2),
                    func_vecFeat32(wordvec1,wordvec2),
                    func_vecFeat33(wordvec1,wordvec2),
                    func_vecFeat34(wordvec1,wordvec2),
                    func_vecFeat36(wordvec1,wordvec2)]

    return function_list

def make_vecFeat_v2(wordvec1,wordvec2):

    function_list= [#func_vecFeat1(s1_token,s2_token,word2vec),
                    func_vecFeat3(wordvec1,wordvec2),
                    func_vecFeat4(wordvec1,wordvec2),
                    func_vecFeat5(wordvec1,wordvec2),
                    func_vecFeat6(wordvec1,wordvec2),
                    func_vecFeat7(wordvec1,wordvec2),
                    func_vecFeat9(wordvec1,wordvec2),
                    func_vecFeat12(wordvec1,wordvec2),
                    func_vecFeat13(wordvec1,wordvec2),
                    func_vecFeat14(wordvec1,wordvec2),
                    func_vecFeat15(wordvec1,wordvec2),
                    func_vecFeat16(wordvec1,wordvec2),
                    func_vecFeat18(wordvec1,wordvec2),
                    func_vecFeat21(wordvec1,wordvec2),
                    func_vecFeat22(wordvec1,wordvec2),
                    func_vecFeat23(wordvec1,wordvec2),
                    func_vecFeat24(wordvec1,wordvec2),
                    func_vecFeat25(wordvec1,wordvec2),
                    func_vecFeat27(wordvec1,wordvec2),
                    func_vecFeat30(wordvec1,wordvec2),
                    func_vecFeat31(wordvec1,wordvec2),
                    func_vecFeat32(wordvec1,wordvec2),
                    func_vecFeat33(wordvec1,wordvec2),
                    func_vecFeat34(wordvec1,wordvec2),
                    func_vecFeat36(wordvec1,wordvec2)]

    return function_list


if __name__=='__main__':
    word2vec = Word2Vec()
    print(make_vecFeat('Obama speaks to the media in Illinois',
                       'The president greets the press in Chicago', word2vec))

