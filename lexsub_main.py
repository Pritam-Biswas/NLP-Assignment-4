#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from nltk.stem import WordNetLemmatizer
from string import punctuation
# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    for syn in wn.synsets(lemma, pos = pos):
        for l in syn.lemmas():
            possible_synonyms.append(l.name())
    possible_synonyms = list(set(possible_synonyms))

    for i in range(0,len(possible_synonyms)):
        if '_' in possible_synonyms[i]:
           possible_synonyms[i] =  possible_synonyms[i].replace('_',' ')

    possible_synonyms = set(filter(lambda a: a != lemma, possible_synonyms))

    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    word_dict = {}
    lemma = context.lemma
    pos = context.pos
    for syn in wn.synsets(lemma, pos = pos):
        for l in syn.lemmas():
            if l.name() == lemma:
                continue
            if l.name() in word_dict:
                word_dict[l.name()] += l.count()
            else:
                word_dict[l.name()] = l.count()

    max_val = 0
    max_key = ''
    for key in word_dict.keys():
        if max_val == 0:
            max_val = word_dict[key]
            max_key = key
        if word_dict[key] > max_val:
            max_val = word_dict[key]
            max_key = key
    # print (word_dict)
    if '_' in max_key:
        max_key = max_key.replace('_',' ')
    return max_key

def wn_simple_lesk_predictor(context):
    lemma = context.lemma
    pos = context.pos
    stop_words = stopwords.words('english')
    res = ''
    overlap_dict = {}


    max_val = 0
    max_key = ''

    for syn in wn.synsets(lemma, pos = pos):
        def_str = syn.definition().strip()
        examples_str = (' ').join(syn.examples()).strip()
        hyper_set = syn.hypernyms()
        hyper_str = ''
        for item in hyper_set:
            hyper_str = hyper_str + ' ' + item.definition() + ' '
            hyper_str = hyper_str + ' ' + (' ').join(item.examples()) + ' '

        res_str = def_str + ' ' + examples_str + ' ' + hyper_str
        context_str = (' ').join(context.left_context) + ' ' + (' ').join(context.right_context)

        res_list = res_str.split(' ')
        context_list = context_str.split(' ')

        res_list = list(set(res_list))
        context_list = list(set(context_list))

        for i in range(0,len(res_list)):
            if not res_list[i].isalpha() or res_list[i] in stop_words:
                res_list[i] = '-1'

        for i in range(0,len(context_list)):
            if not context_list[i].isalpha() or context_list[i] in stop_words:
                context_list[i] = '-1'

        res_list  = list(filter(lambda a: a != '-1', res_list))
        context_list  = list(filter(lambda a: a != '-1', context_list))

        overlap_count = 0
        for res in res_list:
            if res in context_list:
                overlap_count += 1

        if overlap_count > max_val:
            name_flag = 0
            for l in syn.lemmas():
                name = l.name()
                if not name == lemma:
                    name_flag = 1
            if name_flag == 0:
                continue
            else:
                max_val = overlap_count
                max_key = syn

    if not max_val == 0:
        max_count = -1
        max_lemma = ''
        for l in max_key.lemmas():
            name = l.name()
            count = l.count()

            if name == lemma:
                continue
            if count > max_count:
                max_count = count
                max_lemma = name

        return max_lemma

    else:
        max_count = -1
        max_lemma = ''
        for syn in wn.synsets(lemma, pos = pos):
            for l in syn.lemmas():
                name = l.name()
                count = l.count()

                if name == lemma:
                    continue
                if count > max_count:
                    max_count = count
                    max_lemma = name
        return max_lemma
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        lemma = context.lemma
        pos = context.pos

        possible_synonyms = get_candidates(lemma, pos)
        sim_dict = {}
        max_val = -1
        max_key = ''
        for syn in possible_synonyms:
            try:
                score = self.model.similarity(lemma, syn)
            except Exception as e:
                score = 0
            # print (score)
            sim_dict[syn] = score
            if max_val == -1:
                max_val = score
                max_key = syn
            if score > max_val:
                max_val = score
                max_key = syn

        return max_key

    def predict_nearest_with_context(self, context): 
        stop_words = stopwords.words('english')
        lemma = context.lemma
        pos = context.pos

        right = []
        left  = []

        for word in context.right_context:
            if word.isalpha() and not word.lower() in stop_words and not word in punctuation:
                right.append(word)

        for word in context.left_context:
            if word.isalpha() and not word.lower() in stop_words  and not word in punctuation:
                left.append(word)


        res_list = left[-5:] +  right[0:5] + [context.lemma]
 
        res_vec = np.zeros(300)
        for word in res_list:
            try : 
                res_vec = res_vec + self.model.wv[word]
            except : 
                continue
        synonyms = get_candidates(lemma, pos)
        max_score = -1
        max_key = ''
        for syn in synonyms:
            try:
                syn_vec = self.model.wv[syn]
                sim_score = np.dot(syn_vec,res_vec) / (np.linalg.norm(syn_vec)*np.linalg.norm(res_vec))
            except Exception as e:
                continue
            if max_score == -1:
                max_score = sim_score
                max_key = syn
            if sim_score > max_score:
                max_score = sim_score
                max_key = syn
        
        return max_key
    def predict_nearest_custom(self, context):
        stop_words = stopwords.words('english')
        lemma = context.lemma
        pos = context.pos

        right = []
        left  = []

        for word in context.right_context:
            if word.isalpha() and not word.lower() in stop_words and not word in punctuation:
                right.append(word)

        for word in context.left_context:
            if word.isalpha() and not word.lower() in stop_words  and not word in punctuation:
                left.append(word)


        res_list = left[-2:] +  right[0:2] + [context.lemma]
 
        res_vec = np.zeros(300)
        for word in res_list:
            try : 
                res_vec = res_vec + self.model.wv[word]
            except : 
                continue

        def cos_score(i,j):
            return np.dot(i,j) / (np.linalg.norm(i)*np.linalg.norm(j))

        synonyms = get_candidates(lemma, pos)
        max_score = -1
        max_key = ''
        for syn in synonyms:
            try:
                syn_vec = self.model.wv[syn]
                sim_score = cos_score(syn_vec, res_vec)
            except Exception as e:
                continue
            if max_score == -1:
                max_score = sim_score
                max_key = syn
            if sim_score > max_score:
                max_score = sim_score
                max_key = syn
        
        return max_key


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction_f = wn_frequency_predictor(context)
        # prediction_l = wn_simple_lesk_predictor(context)
        # prediction_emb = predictor.predict_nearest(context)
        # prediction_con = predictor.predict_nearest_with_context(context)
        prediction_cust = predictor.predict_nearest_custom(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction_cust))
